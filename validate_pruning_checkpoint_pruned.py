import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from custom_llms.qwen2 import Qwen2ForCausalLM  # Ensure this import is correct
from custom_llms.llama_disp import LlamaForCausalLM
from alignment_function_llm import Group_Lasso_regularization
from sklearn.metrics import precision_recall_fscore_support
from rouge_score import rouge_scorer
import re
import math
import os
from peft import LoftQConfig, LoraConfig, get_peft_model
from util_llm import LoRALinear, customized_lora_substitution
from custom_llms.pruned_llama_disp import PrunedLlamaForCausalLM, model_replace

def transform_output(inputs):
    lw_structure = [128] * 64 + [4096] + [11008]
    arch_vector = []
    start = 0
    for i in range(len(lw_structure)):
        end = start + lw_structure[i]
        arch_vector.append(inputs[:, start:end])
        start = end
    return arch_vector

def transform_output_layer_DISP(inputs, model_name=None):
    if model_name  == 'llama2-7b':
        lw_structure = [128] + [4096] + [4096] + [11008] + [4096]
        num_kv_heads = 32
    elif model_name == 'llama3-8b':
        lw_structure = [128] * 2 + [14336]
        num_kv_heads = 8

    arch_vector = []
    start = 0
    for i, size in enumerate(lw_structure):
        end = start + size
        sliced_input_tensor = inputs[:, start:end]

        if i < 1:  # Extend K_V_head_mask for the whole layer (multi-head)
            replicated_slices = sliced_input_tensor.repeat(1, num_kv_heads)
            arch_vector.append(replicated_slices)
        else:
            arch_vector.append(sliced_input_tensor)
        start = end
    return arch_vector


def initialize_model_and_tokenizer(pruned_ckpt_path=None, model_name=None):
    ckpt_path = pruned_ckpt_path
    print(f"Loading pruned_model_checkpoint from {ckpt_path}.")
    checkpoint    = torch.load(ckpt_path, map_location=torch.device('cpu'))
    masks = checkpoint["pruning_masks"]
    if model_name == 'llama2-7b':
        print("Initializing LLaMA 2-7B model.")
        api_token = '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'
        model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = PrunedLlamaForCausalLM(model_cfg, masks).cuda()
        model.resize_token_embeddings(len(tokenizer))

    model_replace(model, model_name)
    
    customized_lora_substitution(model, rank=32, dropout=0.1)
    print(model)
    print("Loading state dict from checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, tokenizer, masks


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
    prompt = input_ids
    text = input_ids[0]

    with torch.no_grad():
        past_key_values = None  # Initialize past_key_values to None
        input_ids = prompt  # Initial input

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
    text = text[prompt.shape[1]:] 

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
    import argparse
    
    model_name = 'llama2-7b'

    parser = argparse.ArgumentParser(description="Run the model with user-defined checkpoint path")
    parser.add_argument("--pruned_ckpt_path", type=str, required=True, help="Path to the checkpoint file for the pruned model")
    args = parser.parse_args()

    # Use the user-defined checkpoint path
    ckpt_path = args.pruned_ckpt_path

    model, tokenizer, masks = initialize_model_and_tokenizer(pruned_ckpt_path=ckpt_path, model_name=model_name)

    while True:
        dataset_name = input("Enter the dataset to evaluate (PubMedQA/MedNLI/HQS/Harrison) or type 'exit' to quit: ").strip().lower()
        
        if dataset_name == 'exit':
            print("Exiting the evaluation loop.")
            break
        elif dataset_name == 'free':
            general_text_completion(model, tokenizer, masks)
        elif dataset_name == 'instruct':
            evaluate_instruction(model, tokenizer, masks)