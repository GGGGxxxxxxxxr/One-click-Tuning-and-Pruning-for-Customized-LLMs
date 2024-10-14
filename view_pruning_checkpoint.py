import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding
from custom_llms.qwen2 import Qwen2ForCausalLM
from datasets import load_dataset

def transform_output(inputs):
        lw_structure = [64, 64, 64, 64, 896,4864]
        arch_vector = []
        start = 0
        for i in range(len(lw_structure)):
            end = start + lw_structure[i]
            arch_vector.append(inputs[:, start:end])
            start = end

        return arch_vector

checkpoint = torch.load("/home/user1/workspace/leilu/AutoTrainOnce/checkpoint.pth.tar", map_location=torch.device('cpu'))  # adjust map_location as needed
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model     = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").cuda()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

layer0 = model.model.layers[0]
cur_mask_vec = checkpoint["mask_vec"].to("cuda")
mask = transform_output(cur_mask_vec)

# build test/validation dataset
val_set = load_dataset("json", data_files="/home/user1/workspace/leilu/AutoTrainOnce/nlp_dataset_collections/medNLI/mli_test_v1.jsonl").remove_columns(
        ["pairID", "sentence1_parse", "sentence1_binary_parse", "sentence2_parse", "sentence2_binary_parse"]
    )
val_set = val_set["train"]

acc_count = 0

for i in range(len(val_set)):
    sentence1 = val_set[i]["sentence1"]
    sentence2 = val_set[i]["sentence2"]
    gold_label = val_set[i]["gold_label"]

    input_text = f"Predict the #gold_label# from 'entailment', 'contradiction' or 'neutral' based on the content of #sentence1# and #sentence2#. #sentence1#: '{sentence1}', #sentence2#: '{sentence2}'. Predicted #gold_label#:"

    generated_text = input_text
    target_text    = input_text

    for _ in range(1):
        model_inputs = tokenizer([generated_text], return_tensors="pt").to("cuda")
        target_inputs = tokenizer([target_text], return_tensors="pt").to("cuda")
        with torch.autocast(device_type='cuda'):
            model_output = model(input_ids = model_inputs["input_ids"], attention_mask = model_inputs["attention_mask"], return_dict=True)
            target_output = model(input_ids = target_inputs["input_ids"], attention_mask = target_inputs["attention_mask"], return_dict=True, pruning_mask=mask)

        logits = model_output.logits
        t_logits = target_output.logits

        next_token_logits = logits[:, -1, :]
        t_next_token_logits = t_logits[:, -1, :]

        probabilities = torch.softmax(next_token_logits, dim=-1)
        t_prob = torch.softmax(t_next_token_logits, dim=-1)

        # sample next token based on probability
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        t_next_token_id = torch.multinomial(t_prob, num_samples=1)
        # token_id to readable texts
        next_token = tokenizer.decode(next_token_id[0])
        t_next_token = tokenizer.decode(t_next_token_id[0])

        generated_text += next_token
        target_text += t_next_token

    ### judge
    if 'entail' in t_next_token:
        prediction = 'entailment'
    elif 'neutral' in t_next_token:
         prediction = 'neutral'
    elif 'contradiction' in t_next_token:
         prediction = 'contradiction'
    
    if prediction == gold_label:
        acc_count += 1
        print("BINGO!")
    else:
        print("BOOM!")
    #print(f"expected: {gold_label}, predicted: {prediction}")

print(acc_count)