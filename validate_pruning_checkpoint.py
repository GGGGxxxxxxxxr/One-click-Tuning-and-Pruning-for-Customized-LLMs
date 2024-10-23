import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding
from custom_llms.qwen2 import Qwen2ForCausalLM
from custom_llms.llama import LlamaForCausalLM
from datasets import load_dataset
from alignment_function_llm import Group_Lasso_regularization


def transform_output(inputs):
        lw_structure = [128] * 64 + [4096] + [11008]
        arch_vector = []
        start = 0
        for i in range(len(lw_structure)):
            end = start + lw_structure[i]
            arch_vector.append(inputs[:, start:end])
            start = end

        return arch_vector

def transform_output_layer_uniform(inputs):
    lw_structure = [128] * 2 + [4096] + [11008]
    num_kv_heads = 32
    arch_vector = []
    start = 0
    for i, size in enumerate(lw_structure):
        end                 = start + size
        sliced_input_tensor = inputs[:, start : end]

        if i < 2:  # we need to extend K_V_head_mask for the whole layer (multi-head)
            replicated_slices = [sliced_input_tensor] * num_kv_heads
            arch_vector.extend(replicated_slices)
        else:
            arch_vector.append(sliced_input_tensor)
        start = end

    return arch_vector
'''
checkpoint = torch.load("/home/user1/workspace/leilu/AutoTrainOnce/checkpoint.pth.tar", map_location=torch.device('cpu'))  # adjust map_location as needed
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model     = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").cuda()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
layer0 = model.model.layers[0]
cur_mask_vec = checkpoint["mask_vec"].to("cuda")
mask = transform_output(cur_mask_vec)
'''


print("loading checkpoint.")
checkpoint = torch.load("/orange/yonghui.wu/sgao1/llm_pruning_test.pth.tar", map_location=torch.device('cpu'))

print("llama2-7b model initialization.")
api_token = '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'
model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf",  token= api_token)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token = api_token)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="sdpa", token = api_token).cuda()
model.resize_token_embeddings(len(tokenizer))

print("load state dict from ckpt.")
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

print("check weight")
print(model.model.layers[0].mlp.gate_proj.weight)
print(model.model.layers[0].mlp.up_proj.weight)
print(model.model.layers[0].mlp.down_proj.weight)

print("get cur_mask_vec")
cur_mask_vec = checkpoint["mask_vec"].to("cuda")
#masks = transform_output(cur_mask_vec)
masks = transform_output_layer_uniform(cur_mask_vec)

### default: view current pruning pattern
## attention pruning pattern
attn_k_mask = masks[:32]
attn_v_mask = masks[32:64]
attn_out_mask = masks[-2]
attn_k_pruning_dim = [(1-inv_mask).sum(dim=1) for inv_mask in attn_k_mask]
attn_v_pruning_dim = [(1-inv_mask).sum(dim=1) for inv_mask in attn_v_mask]
print(f"attn_k_pruning_pattern: {attn_k_pruning_dim}")
print(f"attn_v_pruning_pattern: {attn_v_pruning_dim}")



### option1: debugging for GroupLasso WeightProjection
print("view pruning pattern.")
for layer_idx in range(32):
      layer_wise_masks = [individual_mask[layer_idx,:] for individual_mask in masks]
      mlp_up_mask = layer_wise_masks[-1]
      print(f"layer_{layer_idx}_mlp_up_mask_shape: {mlp_up_mask.size()}")
      mlp_up_mask_ratio = (1-mlp_up_mask).sum() / mlp_up_mask.numel()
      print(f"layer_{layer_idx}_mlp_up_mask_ratio: {mlp_up_mask_ratio}")

print("validate grouplasso regularization")
gl_loss_module = Group_Lasso_regularization(args = None, target_llm_cfg = model_cfg, prunable_structure = None, fsdp_scaler=None)
gl_loss_module.debug_purpose_compute(target_llm=model, pruning_masks=masks, epoch=None)

### option2: debugging for local weight projection
#status = gl_loss_module.debug_purpose_nofsdp_project_weight(target_llm=model, pruning_masks=masks,epoch=2,lr=1e-4)
#print("After local grouplasso projection:")
#gl_loss_module.debug_purpose_compute(target_llm=model, pruning_masks=masks, epoch=None)


### option3: debugging for real test on pruned model or masked model
# build test/validation dataset

'''
val_set = load_dataset("json", data_files="nlp_dataset_collections/medNLI/mli_test_v1.jsonl").remove_columns(
        ["pairID", "sentence1_parse", "sentence1_binary_parse", "sentence2_parse", "sentence2_binary_parse"]
    )
val_set = val_set["train"]
'''

val_set = load_dataset("json", data_files="nlp_dataset_collections/PubMedQA/pubMedQA_test.jsonl")["train"]
print("dataset perf evaluation...")
input("press ENTER to continue...")

#val_set = load_dataset("fancyzhx/ag_news")["test"]
acc_count_base   = 0
acc_count_masked = 0

for i in range(len(val_set)):
    
    #sentence1 = val_set[i]["sentence1"]
    #sentence2 = val_set[i]["sentence2"]
    #gold_label = val_set[i]["gold_label"]
    
    #sentence = val_set[i]["text"]
    #gold_label = val_set[i]["label"]

    #print(sentence)
    #print(gold_label)

    #input_text = f"Predict the #gold_label# from 'entailment', 'contradiction' or 'neutral' based on the content of #sentence1# and #sentence2#. #sentence1#: '{sentence1}', #sentence2#: '{sentence2}'. Predicted #gold_label#:"
    #input_text = f"Predict the #class_label# from '0', '1', '2' or '3' based on the content of #sentence#. #sentence#: '{sentence}'. Predicted #class_label#: "
    #input_text = (
    #    f"Premise is '{sentence1}', and hypothesis is '{sentence2}'. "
    #    f"Their relationship is ' "
    #)
    context = val_set[i]['CONTEXTS']
    question = val_set[i]['QUESTION']
    gold_label = val_set[i]['final_decision']

    print(gold_label)

    input_text = (
        f"The abstract of a biomedical research article is '{context}'. "
        f"Here comes a question '{question}', and please answer the question with 'yes', 'no', or 'maybe'. "
        f"The answer is '"
    )
    generated_text = input_text
    target_text    = input_text

    prediction = None
    prediction_b = None

    for _ in range(1):
        model_inputs = tokenizer([generated_text], return_tensors="pt").to("cuda")
        target_inputs = tokenizer([target_text], return_tensors="pt").to("cuda")
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_output = model(input_ids = model_inputs["input_ids"], attention_mask = model_inputs["attention_mask"], return_dict=True)
            target_output = model(input_ids = target_inputs["input_ids"], attention_mask = target_inputs["attention_mask"], return_dict=True, pruning_mask=masks)

        logits = model_output.logits
        t_logits = target_output.logits

        next_token_logits = logits[:, -1, :]
        t_next_token_logits = t_logits[:, -1, :]

        probabilities = torch.softmax(next_token_logits, dim=-1)
        t_prob = torch.softmax(t_next_token_logits, dim=-1)

        # sample next token based on probability
        '''
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        t_next_token_id = torch.multinomial(t_prob, num_samples=1)
        '''

        next_token_id = torch.argmax(probabilities, dim=-1)
        t_next_token_id = torch.argmax(t_prob, dim=-1)

        # token_id to readable texts
        next_token = tokenizer.decode(next_token_id[0])
        t_next_token = tokenizer.decode(t_next_token_id[0])

        generated_text += next_token
        target_text += t_next_token

    ### judge
    print(f"next_token_via_sparse_model: {next_token}")
    print(f"next_token_via_mask: {t_next_token}")

    
    if 'ent' in t_next_token:
        prediction = 'entailment'
    elif 'neut' in t_next_token:
         prediction = 'neutral'
    elif 'contr' in t_next_token:
         prediction = 'contradiction'
    
    if 'ent' in next_token:
        prediction_b = 'entailment'
    elif 'neut' in next_token:
         prediction_b = 'neutral'
    elif 'contr' in next_token:
         prediction_b = 'contradiction'

    '''
    if '1' in next_token:
         prediction = 1
    elif '2' in next_token:
         prediction = 2 
    elif '3' in next_token:
         prediction = 3
    elif '0' in next_token:
         prediction = 0

    if '1' in t_next_token:
         prediction_b = 1
    elif '2' in t_next_token:
         prediction_b = 2 
    elif '3' in t_next_token:
         prediction_b = 3
    elif '0' in t_next_token:
         prediction_b = 0
    '''

    if prediction == gold_label:
        acc_count_masked += 1
        print("BINGO!")
    else:
        print("BOOM!")
    
    if prediction_b == gold_label:
         acc_count_base += 1
    #print(f"expected: {gold_label}, predicted: {prediction}")

print(acc_count_base)
print(acc_count_masked)