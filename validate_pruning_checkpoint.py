import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from custom_llms.qwen2 import Qwen2ForCausalLM  # Ensure this import is correct
from custom_llms.llama import LlamaForCausalLM
from alignment_function_llm import Group_Lasso_regularization
from sklearn.metrics import precision_recall_fscore_support
from rouge_score import rouge_scorer
import re
import math
import os
from peft import LoftQConfig, LoraConfig, get_peft_model
from util_llm import LoRALinear, customized_lora_substitution

def transform_output(inputs):
    lw_structure = [128] * 64 + [4096] + [11008]
    arch_vector = []
    start = 0
    for i in range(len(lw_structure)):
        end = start + lw_structure[i]
        arch_vector.append(inputs[:, start:end])
        start = end
    return arch_vector

def transform_output_layer_uniform(inputs, model_name=None):
    if model_name  == 'llama2-7b':
        lw_structure = [128] * 2 + [11008]
        num_kv_heads = 32
    elif model_name == 'llama3-8b':
        lw_structure = [128] * 2 + [14336]
        num_kv_heads = 8

    arch_vector = []
    start = 0
    for i, size in enumerate(lw_structure):
        end = start + size
        sliced_input_tensor = inputs[:, start:end]

        if i < 2:  # Extend K_V_head_mask for the whole layer (multi-head)
            replicated_slices = sliced_input_tensor.repeat(1, num_kv_heads)
            arch_vector.append(replicated_slices)
        else:
            arch_vector.append(sliced_input_tensor)
        start = end
    return arch_vector

def initialize_model_and_tokenizer(base=False, lora=False, input_ckpt_path=None, model_name=None):
    ckpt_path = input_ckpt_path
    print(f"Loading checkpoint from {ckpt_path}.")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    if model_name == 'llama2-7b':
        print("Initializing LLaMA 2-7B model.")
        api_token = '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'
        model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            token=api_token
        ).cuda()
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == 'llama3-8b':
        print("Initializing LLaMA 3-8B model.")
        api_token = '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'
        model_cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B", token=api_token)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=api_token)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            token=api_token
        ).cuda()
        model.resize_token_embeddings(len(tokenizer))


    if lora == True:
        print("intialize LoRA insertions.")
        if base == True:
            print("intialize LoRA insertions.")
            lora_config = LoraConfig(
                                r=8,
                                lora_alpha=8,
                                target_modules="all-linear",
                                lora_dropout=0.1,
                                bias="none"
                            )
            # lora detailed configuration
            print(f"  r: {lora_config.r}")
            print(f"  lora_alpha: {lora_config.lora_alpha}")
            print(f"  target_modules: {lora_config.target_modules}")
            print(f"  lora_dropout: {lora_config.lora_dropout}")
            print(f"  bias: {lora_config.bias}")
            # fuse lora module into pre-trained target llm
            model = get_peft_model(model, lora_config)
        
        else:
            print("use customized lora-blocks.")
            customized_lora_substitution(model, rank=8, dropout=0.1)

    print("Loading state dict from checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    if not base:
        print("Getting current mask vector.")
        cur_mask_vec = checkpoint["mask_vec"].to("cuda")
        masks = transform_output_layer_uniform(cur_mask_vec, model_name=model_name)

        # Include weight mask observation parts
        observe_weight_masks(model, model_cfg, masks)

        return model, tokenizer, masks
    
    return model, tokenizer, None

def observe_weight_masks(model, model_cfg, masks):
    num_key_values = model_cfg.num_key_value_heads
    # Check weights of the first layer's MLP
    print("Checking weights of the first layer's MLP:")
    print("gate_proj.lora_A:")
    print(model.model.layers[0].mlp.gate_proj.lora_A)
    print("gate_proj.lora_B:")
    print(model.model.layers[0].mlp.gate_proj.lora_B)
    print("up_proj.lora_A:")
    print(model.model.layers[0].mlp.up_proj.lora_A)
    print("up_proj.lora_B:")
    print(model.model.layers[0].mlp.up_proj.lora_B)
    print("down_proj.lora_A:")
    print(model.model.layers[0].mlp.down_proj.lora_A)
    print("down_proj.lora_B:")
    print(model.model.layers[0].mlp.down_proj.lora_B)

    '''
    print("up_proj.weight:")
    print(model.model.layers[0].mlp.up_proj.weight)
    print("down_proj.weight:")
    print(model.model.layers[0].mlp.down_proj.weight)
    '''

    # View current pruning pattern
    print("Viewing current pruning pattern.")
    attn_k_mask = masks[0]
    attn_v_mask = masks[1]
    mlp_mask    = masks[2]
    #attn_k_pruning_dim = [(1 - inv_mask).sum(dim=1) for inv_mask in attn_k_mask]
    attn_k_after_pruning = torch.sum(attn_k_mask, dim=1) / num_key_values
    attn_v_after_pruning = torch.sum(attn_v_mask, dim = 1) / num_key_values
    print(attn_v_mask[23])
    #attn_v_pruning_dim = [(1 - inv_mask).sum(dim=1) for inv_mask in attn_v_mask]
    print(f"attn_k_pruning_pattern: {attn_k_after_pruning}")
    print(f"attn_v_pruning_pattern: {attn_v_after_pruning}")
    #print(f"attn_o_pruning_pattern: {attn_o_pruning_dim}")

    # Debugging for Group Lasso Weight Projection
    print("Viewing pruning patterns for each layer.")
    for layer_idx in range(32):
        layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in masks]
        mlp_up_mask = layer_wise_masks[-1]
        print(f"Layer {layer_idx}:")
        print(f"mlp_up_mask_shape: {mlp_up_mask.size()}")
        mlp_up_mask_ratio = (1 - mlp_up_mask).sum() / mlp_up_mask.numel()
        print(f"mlp_up_mask_ratio: {mlp_up_mask_ratio}")

    # compute the current params remaining
    params_Q = torch.sum(torch.sum(attn_k_mask, dim=1) * 4096) * 4
    params_K = torch.sum(torch.sum(attn_k_mask, dim=1) * 4096)
    params_V = torch.sum(torch.sum(attn_v_mask, dim=1) * 4096)
    params_out = torch.sum(torch.sum(attn_v_mask, dim=1) * 4096) * 4
    params_up_gate = torch.sum(torch.sum(mlp_mask, dim=1) * 4096) * 2
    params_down = torch.sum(torch.sum(mlp_mask, dim=1) * 4096)

    total_remain_params = params_Q + params_K + params_V + params_out + params_up_gate + params_down
    print(f"current remaining params: {total_remain_params}")

    '''
    # Validate Group Lasso regularization
    print("Validating Group Lasso regularization.")
    gl_loss_module = Group_Lasso_regularization(
        args=None,
        target_llm_cfg=model_cfg,
        prunable_structure=None,
        fsdp_scaler=None
    )
    gl_loss_module.debug_purpose_compute(
        target_llm=model,
        pruning_masks=masks,
        epoch=None
    )
    '''
def evaluate_model_on_dataset(model, tokenizer, masks, dataset_name):
    if dataset_name.lower() == 'pubmedqa':
        dataset = load_dataset(
            "json",
            data_files="nlp_dataset_collections/PubMedQA/pubMedQA_test.jsonl"
        )["train"]
        evaluate_pubmedqa(model, tokenizer, masks, dataset)
    elif dataset_name.lower() == 'mednli':
        dataset = load_dataset(
            "json",
            data_files="nlp_dataset_collections/medNLI/mli_test_v1.jsonl"
        ).remove_columns(
            ["pairID", "sentence1_parse", "sentence1_binary_parse", "sentence2_parse", "sentence2_binary_parse"]
        )["train"]
        evaluate_mednli(model, tokenizer, masks, dataset)
    elif dataset_name.lower() == 'hqs':
        dataset = load_dataset(
            "json",
            data_files="nlp_dataset_collections/HQS/HQS_test.jsonl"
        )["train"]
        evaluate_healthquestionsum(model, tokenizer, dataset, masks)
    elif dataset_name.lower() == 'harrison':
        evaluate_perplexity_on_harrison(model, tokenizer, masks)
    elif dataset_name.lower() == 'multilegalpile':
        evaluate_perplexity_on_multilegalpile(model, tokenizer, masks)
    elif dataset_name.lower() == 'casehold':
        evaluate_casehold(model, tokenizer, masks)
    elif dataset_name.lower() == 'billsum':
        evaluate_billsum(model, tokenizer, masks)
    else:
        print(f"Dataset '{dataset_name}' is not supported.")
        return

def evaluate_pubmedqa(model, tokenizer, masks, dataset):
    print("Evaluating on PubMedQA dataset...")
    true_labels = []
    pred_labels = []

    for i in range(len(dataset)):
        context = " ".join(dataset[i]['CONTEXTS'])
        question = dataset[i]['QUESTION']
        gold_label = dataset[i]['final_decision'].lower()

        input_text = (
            f"The abstract of a biomedical research article is '{context}'. "
            f"Here comes a question '{question}', and please answer the question with 'yes', 'no', or 'maybe'. "
            f"The answer is '"
        )

        prediction = generate_predictions(model, tokenizer, input_text, masks)

        # Map prediction to one of the labels
        prediction = prediction.lower()
        if "yes" in prediction:
            prediction = 'yes'
        elif 'maybe' in prediction or 'ma' in prediction:
            prediction = 'maybe'
        elif 'no' in prediction:
            prediction = 'no'
        else:
            prediction = 'unknown'  # For unexpected predictions

        true_labels.append(gold_label)
        pred_labels.append(prediction)

        print(f"Sample {i+1}/{len(dataset)} | Gold: {gold_label} | Prediction: {prediction}")

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=['yes', 'no', 'maybe'], average=None, zero_division=0
    )

    # Calculate macro-F1 score
    macro_f1 = f1.mean()

    # Print per-class metrics
    for i, label in enumerate(['yes', 'no', 'maybe']):
        print(f"Class '{label}': Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {f1[i]:.4f}, Support: {support[i]}")

    print(f"\nMacro-F1 Score: {macro_f1:.4f}")



def evaluate_mednli(model, tokenizer, masks, dataset):
    print("Evaluating on MedNLI dataset...")
    acc_count_base = 0

    for i in range(len(dataset)):
        sentence1 = dataset[i]["sentence1"]
        sentence2 = dataset[i]["sentence2"]
        gold_label = dataset[i]["gold_label"]

        input_text = (
            f"Premise is '{sentence1} and hypothesis is '{sentence2}."
            f"Their relationship is '"
        )
        
        prediction_base = generate_predictions(model, tokenizer, input_text, masks)
        #generated_text = generate_summary(model, tokenizer, input_text, masks, True, max_length=10)
        
        if "contr" in prediction_base:
            prediction_base = "contradiction"
        elif "ent" in prediction_base:
            prediction_base = "entailment"
        elif "neu" in prediction_base:
            prediction_base = "neutral"
        else:
            prediction_base = None

        if prediction_base == gold_label:
            acc_count_base += 1


        print(f"Sample {i+1}/{len(dataset)} | Gold: {gold_label} | Base Prediction: {prediction_base}")

    print(f"Pruned Model Accuracy: {acc_count_base / len(dataset) * 100:.2f}%")


def evaluate_casehold(model, tokenizer, masks):
    #dataset_file = 'nlp_dataset_collections/CaseHold/casehold_train_clean_2000.jsonl'
    #dataset = load_dataset('json', data_files=dataset_file, split='train')
    dataset = load_dataset("casehold/casehold", "all")['test']

    true_labels = []
    pred_labels = []

    for i in range(200):
        citing_prompt = dataset[i]['citing_prompt']
        holding_statements = [
            dataset[i].get(f'holding_{i}', '') for i in range(5)
        ]
        label = dataset[i]['label']
        
        # 确定索引名称
        idx_mapping = {
            "0": "first",
            "1": "second",
            "2": "third",
            "3": "fourth",
            "4": "fifth"
        }
        idx = idx_mapping.get(str(label), None)
        if idx is None:
            raise ValueError("Label out of expected range.")

        # 根据模板格式化文本
        input_text = (
            f"A citing text consisting of the context and legal citation text is '{citing_prompt}'. "
            f"Holding statement 0 is '{holding_statements[0]}', "
            f"holding statement 1 is '{holding_statements[1]}', "
            f"holding statement 2 is '{holding_statements[2]}', "
            f"holding statement 3 is '{holding_statements[3]}', "
            f"and holding statement 4 is '{holding_statements[4]}'. "
            f"The correct answer is holding statement "
        )

        #prediction = generate_predictions(model, tokenizer, input_text, masks)
        prediction = generate_summary(model, tokenizer, input_text, masks)
        # Map prediction to one of the labels

        if '0' in prediction:
            prediction = '0'
        elif '1' in prediction:
            prediction = '1'
        elif '2' in prediction:
            prediction = '2'
        elif '3' in prediction:
            prediction = '3'
        elif '4' in prediction:
            prediction = '4'

        true_labels.append(label)
        pred_labels.append(prediction)

        print(f"Sample {i+1}/{len(dataset)} | Gold: {label} | Prediction: {prediction}")

    
    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=['0', '1', '2', '3', '4'], average=None, zero_division=0
    )

    # Calculate macro-F1 score
    macro_f1 = f1.mean()

    # Print per-class metrics
    for i, label in enumerate(['0', '1', '2', '3', '4']):
        print(f"Class '{label}': Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {f1[i]:.4f}, Support: {support[i]}")

    print(f"\nMacro-F1 Score: {macro_f1:.4f}")
    


def extract_message(text):
    match = re.search(r'MESSAGE:(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()


def evaluate_healthquestionsum(model, tokenizer, dataset, masks):
    print("Evaluating on HealthQuestionSum dataset...")
    references = []
    hypotheses = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i in range(len(dataset)):
        original_question = dataset[i]['CHQ']
        reference_summary = dataset[i]['Summary']

        question = extract_message(original_question)

        input_text = (
            f"A question posted by a patient is '{question}'."
            f"The summary of the patient's question is: '"
        )

        generated_summary = generate_summary(model, tokenizer, input_text, masks, False, 70)

        references.append(reference_summary)
        hypotheses.append(generated_summary)

        print(f"Sample {i+1}/{len(dataset)}")
        print(f"Question: {question}")
        print(f"Reference Summary: {reference_summary}")
        print(f"Generated Summary: {generated_summary}")
        print("-" * 50)

    # Calculate ROUGE scores
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # Calculate average scores
    for key in rouge_scores:
        avg_score = sum(rouge_scores[key]) / len(rouge_scores[key]) * 100  # Convert to percentage
        print(f"Average {key} F1 Score: {avg_score:.2f}%")

def evaluate_billsum(model, tokenizer, masks):
    print("Evaluating on BillSum dataset...")
    dataset = load_dataset('json', data_files='nlp_dataset_collections/BillSum/billsum_test_200.jsonl', split='train')

    references = []
    hypotheses = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i in range(len(dataset)):
        example = dataset[i]
        reference_summary = example['summary']

        input_text = (
        f"A bill text is '{example['source']}'. "
        f"The summary of the bill is"
        )

        generated_summary = generate_summary(model, tokenizer, input_text, masks, False, 600)

        references.append(reference_summary)
        hypotheses.append(generated_summary)

        print(f"Sample {i+1}/{len(dataset)}")
        print(f"Question: {example['source']}\n")
        print(f"Reference Summary: {reference_summary}\n")
        print(f"Generated Summary: {generated_summary}\n")
        print("-" * 50)

    # Calculate ROUGE scores
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # Calculate average scores
    for key in rouge_scores:
        avg_score = sum(rouge_scores[key]) / len(rouge_scores[key]) * 100  # Convert to percentage
        print(f"Average {key} F1 Score: {avg_score:.2f}%")

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate_text_custom(model, tokenizer, input_ids, max_length=50, masks=None, free=False, top_k=50, top_p=0.9, temperature=0.9):
    model.eval()
    generated = input_ids
    text = input_ids[0]

    with torch.no_grad():
        past_key_values = None  # Initialize past_key_values to None
        input_ids = generated  # Initial input

        for _ in range(max_length):
            if masks is None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, pruning_mask=masks)

            # Get the logits for the last generated token
            next_token_logits = outputs.logits[0, -1, :]

            # Update past_key_values for the next iteration
            past_key_values = outputs.past_key_values

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # Apply softmax to get probabilities
            next_token_probs = torch.softmax(filtered_logits, dim=-1)
        
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)

            # Append the generated token to the sequence
            text = torch.cat((text, next_token_id), dim=0)

            # Update input_ids to only include the newly generated token for the next iteration
            input_ids = next_token_id.unsqueeze(0)

            # Check if the generated token is the EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # remove the prompt_ids 
    text = text[input_ids.shape[1]:] 

    return text

def generate_summary(model, tokenizer, input_text, masks, free=False, max_length=500):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

    generated_ids = generate_text_custom(
        model, tokenizer, input_ids, max_length=max_length, masks=masks, free=free  # 根据需要调整 max_length
    )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    '''
    # 提取摘要
    if generated_text.startswith(input_text):
        generated_summary = generated_text[len(input_text):].strip()
    else:
        generated_summary = generated_text.strip()
    '''

    return generated_text

def generate_predictions(model, tokenizer, input_text, masks):
    model.eval()
    generated_text = input_text

    model_inputs = tokenizer([generated_text], return_tensors="pt").to("cuda")

    # Base Sparse model prediction
    if masks == None:
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            model_output = model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                return_dict=True
            )
    # masked model prediction
    else:
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            model_output = model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                return_dict=True,
                pruning_mask = masks,
            )

    logits = model_output.logits
    next_token_logits = logits[:, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get next token predictions
    next_token_id = torch.argmax(probabilities, dim=-1)
    next_token = tokenizer.decode(next_token_id[0])

    return next_token 

def evaluate_perplexity_on_harrison(model, tokenizer, masks):
    print("Evaluating perplexity on Harrison dataset...")

    # 直接从 harrison.jsonl 文件加载数据
    dataset_file = "nlp_dataset_collections/internalMed/internalMed_test.jsonl"  # 请替换为实际路径

    # 使用 datasets 库加载数据集
    dataset = load_dataset('json', data_files=dataset_file, split='train')

    # 计算困惑度
    perplexity = compute_perplexity(model, tokenizer, dataset, masks)
    print(f"Perplexity on Harrison dataset: {perplexity:.2f}")


def evaluate_perplexity_on_multilegalpile(model, tokenizer, masks):
    print("Evaluating perplexity on MultiLegalPile Dataset...")

    dataset_file = 'nlp_dataset_collections/MultiLegalPile/multilegalpile_300.jsonl'
    dataset = load_dataset('json', data_files=dataset_file, split='train')
    perplexity = compute_perplexity(model, tokenizer, dataset, masks)
    print(f"Perplexity on Harrison dataset: {perplexity:.2f}")


def compute_perplexity(model, tokenizer, dataset, masks):
    total_loss = 0.0
    total_length = 0

    model.eval()
    for example in dataset:
        with torch.no_grad():
            inputs = tokenizer(
                example['text'],
                return_tensors='pt',
                truncation=True,
                #max_length=2048  # 根据需要调整 max_length
            ).to('cuda')

            if masks == None:
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['input_ids']
                    )
            else:
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['input_ids'],
                        pruning_mask = masks
                    )

            loss = outputs.loss
            # 乘以标记数获取总损失
            total_loss   += loss.item() * inputs['input_ids'].size(1)
            total_length += inputs['input_ids'].size(1)

    perplexity = math.exp(total_loss / total_length)
    return perplexity

#-----------------------------------------------------------------#
# 定义通用文本续写函数
def general_text_completion(model, tokenizer, masks):
    """
    处理通用文本续写任务。
    
    参数:
        model: 预训练的语言模型。
        tokenizer: 对应的分词器。
    
    返回:
        None
    """
    print("\n--- 通用文本续写模式 ---")
    print("请输入您的文本（输入 'exit' 退出续写模式）：")
    
    while True:
        user_input = input(">>> ")
        if user_input.lower() == 'exit':
            print("退出通用文本续写模式。\n")
            break
        elif not user_input.strip():
            print("输入为空，请重新输入。")
            continue
        
        # 生成续写
        generated_text = generate_summary(model, tokenizer, user_input, masks, True)
        print(f"续写内容:\n{generated_text}\n")

def evaluate_instruction(model, tokenizer, masks):
    """
    评估模型，用户输入 instruction 和 optional_input，格式化输入并生成响应。

    参数:
        model: 预训练的语言模型。
        tokenizer: 对应的分词器。
        masks: 生成时需要的掩码或其他参数。

    返回:
        None
    """
    print("\n--- 指令评估模式 ---")
    print("请输入您的指令（输入 'exit' 退出）：")

    while True:
        instruction = input("Instruction: ")
        if instruction.lower() == 'exit':
            print("退出指令评估模式。\n")
            break
        elif not instruction.strip():
            print("指令为空，请重新输入。")
            continue

        optional_input = input("Optional Input（若无请直接按 Enter）: ")

        # 根据训练格式格式化输入文本
        if not optional_input.strip():
            input_text = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )
        else:
            input_text = (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{optional_input}\n\n"
                f"### Response:\n"
            )

        generated_text = generate_summary(model, tokenizer, input_text, masks, True)

        # 提取 '### Response:\n' 之后的内容作为模型的响应
        response_start = generated_text.find("### Response:\n")
        if response_start != -1:
            response = generated_text[response_start + len("### Response:\n"):].strip()
        else:
            response = generated_text.strip()

        print(f"Answer:\n{response}\n")

if __name__ == "__main__":
    base = False
    lora = True
    model_name = 'llama3-8b'
    if base != True:
        ckpt_path = "/orange/yonghui.wu/sgao1/llm_pruning_tuning_lora_qa.pth.tar"
    else:
        ckpt_path = "/orange/yonghui.wu/sgao1/llm_base_lora.pth.tar"

    model, tokenizer, masks = initialize_model_and_tokenizer(base=base, lora=lora, input_ckpt_path=ckpt_path, model_name=model_name)

    while True:
        dataset_name = input("Enter the dataset to evaluate (PubMedQA/MedNLI/HQS/Harrison) or type 'exit' to quit: ").strip().lower()
        
        if dataset_name == 'exit':
            print("Exiting the evaluation loop.")
            break
        elif dataset_name == 'free':
            general_text_completion(model, tokenizer, masks)
        elif dataset_name == 'instruct':
            evaluate_instruction(model, tokenizer, masks)
        else:
            evaluate_model_on_dataset(model, tokenizer, masks, dataset_name)
            print("-" * 50)  # 分隔线，便于阅读输出