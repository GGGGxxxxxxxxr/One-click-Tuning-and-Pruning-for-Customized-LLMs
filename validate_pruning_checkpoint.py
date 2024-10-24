import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset, concatenate_datasets
from custom_llms.qwen2 import Qwen2ForCausalLM  # 确保此导入正确
from custom_llms.llama import LlamaForCausalLM
from alignment_function_llm import Group_Lasso_regularization
from sklearn.metrics import precision_recall_fscore_support
from rouge_score import rouge_scorer
import re
import math
import json
import os

#-----------------------------------------------------------------#
# 数据集格式化函数

# MedNLI 数据集格式化
def format_mednli_example(example):
    # 提取必要字段
    sentence1 = example['sentence1']
    sentence2 = example['sentence2']
    gold_label = example['gold_label']
    
    # 根据 gold_label 确定 trailing 文本
    if gold_label == "entailment":
        trailing = "the hypothesis is true given the premise."
    elif gold_label == "contradiction":
        trailing = "the hypothesis is false given the premise."
    elif gold_label == "neutral":
        trailing = "the hypothesis is undetermined given the premise."
    else:
        trailing = "the relationship is unknown."
    
    # 根据提供的模板格式化文本
    formatted_text = (
        f"Premise is '{sentence1}', and hypothesis is '{sentence2}'. "
        f"Their relationship is '{gold_label}', and this means {trailing}"
    )
    
    # 返回包含新字段的字典
    return {'text': formatted_text}

def formatted_MedNLI_dataset(
    train_data_file='nlp_dataset_collections/medNLI/mli_train_v1.jsonl',
    val_data_file='nlp_dataset_collections/medNLI/mli_dev_v1.jsonl',
    num_samples=None
):
    # 加载数据集
    train_set = load_dataset("json", data_files=train_data_file)['train']
    val_set   = load_dataset("json", data_files=val_data_file)['train']
    
    # 移除不必要的列
    columns_to_remove = [
        "pairID", "sentence1_parse", "sentence1_binary_parse",
        "sentence2_parse", "sentence2_binary_parse"
    ]
    train_set = train_set.remove_columns(columns_to_remove)
    val_set   = val_set.remove_columns(columns_to_remove)
    
    # 应用格式化函数并移除原始列
    train_set = train_set.map(
        format_mednli_example,
        remove_columns=["sentence1", "sentence2", "gold_label"]
    )
    val_set   = val_set.map(
        format_mednli_example,
        remove_columns=["sentence1", "sentence2", "gold_label"]
    )
    
    # 如果指定了 num_samples，选择前 num_samples 条数据
    if num_samples is not None:
        num_samples = min(num_samples, len(train_set))
        train_set = train_set.select(range(num_samples))
    
    return train_set, val_set

# PubMedQA 数据集格式化
def format_pubmedqa_example(example):
    # 提取必要字段
    context = example['context']
    question = example['question']
    final_decision = example['final_decision']
    
    # 根据 final_decision 确定 trailing 文本
    if final_decision == "yes":
        trailing = "the phenomenon mentioned by the question is confirmed by the abstract."
    elif final_decision == "no":
        trailing = "we do not support the phenomenon mentioned by the question based on the abstract."
    elif final_decision == "maybe":
        trailing = "we are uncertain whether the phenomenon mentioned by the question is supported by the abstract."
    else:
        trailing = "the answer is unknown."
    
    # 根据提供的模板格式化文本
    formatted_text = (
        f"The abstract of a biomedical research article is '{context}'. "
        f"Here comes a question '{question}', and please answer the question with 'yes', 'no', or 'maybe'. "
        f"The answer is '{final_decision}', which indicates {trailing}"
    )
    
    # 返回包含新字段的字典
    return {'text': formatted_text}

def formatted_PubMedQA_dataset(num_samples=None):
    # 从 Hugging Face 加载 PubMedQA 数据集
    training_dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")['train'].remove_columns(["pubid", "long_answer"])
    validation_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")['train'].remove_columns(["pubid", "long_answer"])
    
    # 如果指定了 num_samples，选择前 num_samples 条数据作为训练集
    if num_samples is not None:
        num_samples = min(num_samples, len(training_dataset))
        training_dataset = training_dataset.select(range(num_samples))
    
    # 对训练集应用格式化函数并移除原始列
    training_dataset = training_dataset.map(
        format_pubmedqa_example,
        remove_columns=["context", "question", "final_decision"]
    )
    
    # 对验证集应用格式化函数并移除原始列
    validation_dataset = validation_dataset.map(
        format_pubmedqa_example,
        remove_columns=["context", "question", "final_decision"]
    )
    
    return training_dataset, validation_dataset

# HealthQuestionSum 数据集格式化
def extract_question(text):
    """
    从文本中提取问题，处理不同的格式。
    1. 如果文本以 'Q. ' 开头，则提取 'Q. ' 后的内容。
    2. 如果文本包含 'SUBJECT:' 或 'MESSAGE:'，则提取 'MESSAGE:' 后的内容。
    3. 否则，将整个文本视为问题。
    
    参数:
        text (str): 原始文本。
        
    返回:
        str: 提取后的问题。
    """
    text = text.strip()
    
    # 1. 检查是否以 'Q. ' 开头
    if text.startswith('Q. '):
        question = text[3:].strip()
        return question
    
    # 2. 检查是否包含 'SUBJECT:' 或 'MESSAGE:'
    if 'SUBJECT:' in text or 'MESSAGE:' in text:
        # 使用正则表达式提取 'MESSAGE:' 后的内容
        match = re.search(r'MESSAGE:\s*(.*)', text, re.DOTALL)
        if match:
            question = match.group(1).strip()
            return question
    
    # 3. 否则，返回整个文本
    return text

def format_hqs_example(example):
    """
    根据指定模板格式化 HealthQuestionSum 数据集的样本。
    
    参数:
        example (dict): 数据集中的单个样本。
        
    返回:
        dict: 包含格式化文本的字典。
    """
    # 提取问题
    question = extract_question(example['CHQ'])
    summary = example['Summary']
    
    # 根据模板格式化文本
    formatted_text = (
        f"A question posted by a patient is '{question}'. "
        f"The summary of the question is '{summary}'."
    )
    
    # 返回包含新字段的字典
    return {'text': formatted_text}

def formatted_HQS_dataset(num_samples=None):
    # 加载数据集并移除不需要的列
    training_dataset   = load_dataset("bigbio/meqsum", "meqsum_source")["train"].remove_columns(["File"])
    validation_dataset = load_dataset("json", data_files="nlp_dataset_collections/HQS/HQS_test.json")['train'].remove_columns("q_id")
    # 如果指定了 num_samples，选择前 num_samples 条数据
    if num_samples is not None:
        num_samples = min(num_samples, len(training_dataset))
        training_dataset = training_dataset.select(range(num_samples))
    
    # 应用格式化函数
    training_dataset = training_dataset.map(format_hqs_example).remove_columns(["CHQ","Summary"])
    validation_dataset = validation_dataset.map(format_hqs_example).remove_columns(["CHQ","Summary"])
    return training_dataset, validation_dataset

# ME_Q_SUM 数据集格式化
def format_me_q_sum_example(example):
    """
    根据指定模板格式化 me_q_sum 数据集的样本。
    
    参数:
        example (dict): 数据集中的单个样本。
        
    返回:
        dict: 包含格式化文本的字典。
    """
    question = example.get('CHQ', '').strip()
    summary = example.get('Summary', '').strip()
    
    # 根据模板格式化文本
    formatted_text = (
        f"A question posted by a patient is '{question}'. "
        f"The summary of the question is '{summary}'."
    )
    
    return {'text': formatted_text}

def rename_me_q_sum_columns(dataset):
    """
    重命名 me_q_sum 数据集的列名，从 'query' 改为 'CHQ'，'answer' 改为 'Summary'。
    
    参数:
        dataset (datasets.Dataset): 原始数据集。
        
    返回:
        datasets.Dataset: 列名已重命名的数据集。
    """
    renamed_dataset = dataset.rename_columns({
        'query': 'CHQ',
        'answer': 'Summary'
    })
    return renamed_dataset

def load_me_q_sum_dataset():
    """
    加载 lighteval/me_q_sum 数据集，并合并 train, validation, test 分区。
    
    返回:
        datasets.Dataset: 合并后的数据集。
    """
    dataset = load_dataset("lighteval/me_q_sum")
    
    # 检查数据集的分区
    print(dataset)
    
    # 合并 train, validation, test 分区
    combined_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    
    return combined_dataset

def formatted_ME_Q_SUM_dataset(num_samples=None):
    """
    加载、重命名并格式化 lighteval/me_q_sum 数据集。
    
    参数:
        num_samples (int, optional): 要选择的样本数量。默认为 None（使用全部数据）。
    
    返回:
        datasets.Dataset: 格式化后的数据集。
    """
    # 加载并合并 train, validation, test 分区
    combined_dataset = load_me_q_sum_dataset()
    
    # 重命名列名
    combined_dataset = rename_me_q_sum_columns(combined_dataset)
    
    # 如果指定了 num_samples，选择前 num_samples 条数据
    if num_samples is not None:
        num_samples = min(num_samples, len(combined_dataset))
        combined_dataset = combined_dataset.select(range(num_samples))
    
    # 应用格式化函数
    formatted_dataset = combined_dataset.map(format_me_q_sum_example).remove_columns(['CHQ', 'Summary'])
    
    return formatted_dataset

#-----------------------------------------------------------------#
# 生成 perplexity_dataset.jsonl 和 intermedMed_train.jsonl
def split_harrison_dataset(original_file='/Users/leilu/Desktop/ATO_llm/nlp_dataset_collections/InternalMed_Harrison.txt',
                           perplexity_file='perplexity_dataset.jsonl',
                           train_file='intermedMed_train.jsonl',
                           perplexity_limit=300):
    """
    从 InternalMed_Harrison.txt 文件中提取前 300 行作为 perplexity_dataset.jsonl，
    其余部分作为 intermedMed_train.jsonl。

    参数:
        original_file (str): 原始文本文件路径。
        perplexity_file (str): 输出 perplexity 数据集的 JSON Lines 文件路径。
        train_file (str): 输出训练数据集的 JSON Lines 文件路径。
        perplexity_limit (int): 用于 perplexity 的行数限制。

    返回:
        None
    """
    if not os.path.exists(original_file):
        print(f"File '{original_file}' not found.")
        return
    
    with open(original_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    # 获取前 300 行
    lines_300 = lines[:perplexity_limit]
    # 准备困惑度数据集
    data_perplexity = [{'text': line} for line in lines_300]
    
    # 写入 perplexity_dataset.jsonl
    with open(perplexity_file, 'w', encoding='utf-8') as outfile:
        for entry in data_perplexity:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    # 获取剩余的行（从第 301 行开始）
    lines_rest = lines[perplexity_limit:]
    # 准备训练数据集
    data_train = [{'text': line} for line in lines_rest]
    
    # 写入 intermedMed_train.jsonl
    with open(train_file, 'w', encoding='utf-8') as outfile:
        for entry in data_train:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    print(f"Split completed. {len(lines_300)} lines for perplexity and {len(lines_rest)} lines for training.")

#-----------------------------------------------------------------#
# Perplexity 数据集加载
def load_intermedMed_train_dataset(train_data_file='intermedMed_train.jsonl'):
    """
    加载 intermedMed_train.jsonl 数据集。

    参数:
        train_data_file (str): intermedMed_train.jsonl 文件的路径。

    返回:
        datasets.Dataset: 加载后的训练数据集。
    """
    if not os.path.exists(train_data_file):
        print(f"File '{train_data_file}' not found.")
        return None

    dataset = load_dataset('json', data_files=train_data_file, split='train')
    return dataset

#-----------------------------------------------------------------#
# 创建并合并所有医学数据集
def create_medical_dataset():
    # 获取各个数据集的训练集和验证集
    mednli_train, mednli_val = formatted_MedNLI_dataset(num_samples=7000)
    pubmedqa_train, pubmedqa_val = formatted_PubMedQA_dataset(num_samples=7000)
    hqs_train, hqs_val = formatted_HQS_dataset(num_samples=1000)
    me_q_sum_train = formatted_ME_Q_SUM_dataset(num_samples=5000)  # 根据需要调整 num_samples
    
    # 加载 intermedMed_train.jsonl 数据集
    intermed_train = load_intermedMed_train_dataset(train_data_file='intermedMed_train.jsonl')
    
    # 合并训练集，包括 intermedMed_train 和 me_q_sum_train
    if intermed_train is not None:
        combined_train = concatenate_datasets([mednli_train, pubmedqa_train, hqs_train, me_q_sum_train, intermed_train])
    else:
        combined_train = concatenate_datasets([mednli_train, pubmedqa_train, hqs_train, me_q_sum_train])
    
    # 合并验证集
    combined_val = concatenate_datasets([mednli_val, pubmedqa_val, hqs_val])
    
    return combined_train, combined_val

#-----------------------------------------------------------------#
# 评估模型功能
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
            data_files="nlp_dataset_collections/HQS/HQS_test.json"
        )["train"]
        evaluate_healthquestionsum(model, tokenizer, dataset)
    elif dataset_name.lower() == 'harrison':
        evaluate_perplexity_on_harrison(model, tokenizer, masks)
    elif dataset_name.lower() == 'free':
        general_text_completion(model, tokenizer)
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

        prediction = generate_predictions(model, tokenizer, input_text)

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
            f"Premise: '{sentence1}'\n"
            f"Hypothesis: '{sentence2}'\n"
            f"Based on the premise, is the hypothesis 'entailment', 'contradiction', or 'neutral'? The answer is '"
        )

        prediction_base = generate_predictions(model, tokenizer, input_text)

        if "cont" in prediction_base:
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

def extract_message(text):
    match = re.search(r'MESSAGE:(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

def evaluate_healthquestionsum(model, tokenizer, dataset):
    print("Evaluating on HealthQuestionSum dataset...")
    references = []
    hypotheses = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i in range(len(dataset)):
        original_question = dataset[i]['CHQ']
        reference_summary = dataset[i]['Summary']

        question = extract_message(original_question)

        input_text = (
            f"A question posted by a patient is '{question}'. "
            f"The summary of the question is '"
        )

        generated_summary = generate_summary(model, tokenizer, input_text)

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

def generate_text_custom(model, tokenizer, input_ids, max_length=50):
    model.eval()
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :]

            # 使用贪心解码
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # 添加下一个标记
            generated = torch.cat((generated, next_token_id), dim=1)

            next_token = tokenizer.decode(next_token_id.squeeze())
            # 检查是否生成了句号或问号
            if next_token == '?' or next_token.strip() == '.':
                break

            # 检查结束标记
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return generated

def generate_summary(model, tokenizer, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

    generated_ids = generate_text_custom(
        model, tokenizer, input_ids, max_length=50  # 根据需要调整 max_length
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 提取摘要
    if generated_text.startswith(input_text):
        generated_summary = generated_text[len(input_text):].strip()
    else:
        generated_summary = generated_text.strip()

    # 移除引号
    if generated_summary.endswith("'"):
        generated_summary = generated_summary[:-1].strip()
    if generated_summary.startswith("'"):
        generated_summary = generated_summary[1:].strip()

    return generated_summary

def generate_predictions(model, tokenizer, input_text):
    generated_text = input_text

    model_inputs = tokenizer([generated_text], return_tensors="pt").to("cuda")

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Base model prediction
        model_output = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            return_dict=True
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
def general_text_completion(model, tokenizer):
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
        generated_text = generate_summary(model, tokenizer, user_input)
        print(f"续写内容:\n{generated_text}\n")

#-----------------------------------------------------------------#
# 初始化模型和标记器
def initialize_model_and_tokenizer():
    print("Loading checkpoint.")
    checkpoint = torch.load("/orange/yonghui.wu/sgao1/llm_pruning_test.pth.tar", map_location=torch.device('cpu'))

    print("Initializing LLaMA 2-7B model.")
    api_token = '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'  # 请确保替换为您的 Hugging Face API Token
    model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=api_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=api_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        attn_implementation="sdpa",
        use_auth_token=api_token
    ).cuda()
    model.resize_token_embeddings(len(tokenizer))

    print("Loading state dict from checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print("Getting current mask vector.")
    cur_mask_vec = checkpoint["mask_vec"].to("cuda")
    masks = transform_output_layer_uniform(cur_mask_vec)

    # Include weight mask observation parts
    observe_weight_masks(model, model_cfg, masks)

    return model, tokenizer, masks

def observe_weight_masks(model, model_cfg, masks):
    # Check weights of the first layer's MLP
    print("Checking weights of the first layer's MLP:")
    print("gate_proj.weight:")
    print(model.model.layers[0].mlp.gate_proj.weight)
    print("up_proj.weight:")
    print(model.model.layers[0].mlp.up_proj.weight)
    print("down_proj.weight:")
    print(model.model.layers[0].mlp.down_proj.weight)

    # View current pruning pattern
    print("Viewing current pruning pattern.")
    attn_k_mask = masks[:32]
    attn_v_mask = masks[32:64]
    attn_out_mask = masks[-2]
    attn_k_pruning_dim = [(1 - inv_mask).sum(dim=1) for inv_mask in attn_k_mask]
    attn_v_pruning_dim = [(1 - inv_mask).sum(dim=1) for inv_mask in attn_v_mask]
    print(f"attn_k_pruning_pattern: {attn_k_pruning_dim}")
    print(f"attn_v_pruning_pattern: {attn_v_pruning_dim}")

    # Debugging for Group Lasso Weight Projection
    print("Viewing pruning patterns for each layer.")
    for layer_idx in range(32):
        layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in masks]
        mlp_up_mask = layer_wise_masks[-1]
        print(f"Layer {layer_idx}:")
        print(f"  mlp_up_mask_shape: {mlp_up_mask.size()}")
        mlp_up_mask_ratio = (1 - mlp_up_mask).sum() / mlp_up_mask.numel()
        print(f"  mlp_up_mask_ratio: {mlp_up_mask_ratio}")

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

#-----------------------------------------------------------------#
# 生成 perplexity_dataset.jsonl 和 intermedMed_train.jsonl
def split_harrison_dataset(original_file='/Users/leilu/Desktop/ATO_llm/nlp_dataset_collections/InternalMed_Harrison.txt',
                           perplexity_file='perplexity_dataset.jsonl',
                           train_file='intermedMed_train.jsonl',
                           perplexity_limit=300):
    """
    从 InternalMed_Harrison.txt 文件中提取前 300 行作为 perplexity_dataset.jsonl，
    其余部分作为 intermedMed_train.jsonl。

    参数:
        original_file (str): 原始文本文件路径。
        perplexity_file (str): 输出 perplexity 数据集的 JSON Lines 文件路径。
        train_file (str): 输出训练数据集的 JSON Lines 文件路径。
        perplexity_limit (int): 用于 perplexity 的行数限制。

    返回:
        None
    """
    if not os.path.exists(original_file):
        print(f"File '{original_file}' not found.")
        return
    
    with open(original_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    # 获取前 300 行
    lines_300 = lines[:perplexity_limit]
    # 准备困惑度数据集
    data_perplexity = [{'text': line} for line in lines_300]
    
    # 写入 perplexity_dataset.jsonl
    with open(perplexity_file, 'w', encoding='utf-8') as outfile:
        for entry in data_perplexity:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    # 获取剩余的行（从第 301 行开始）
    lines_rest = lines[perplexity_limit:]
    # 准备训练数据集
    data_train = [{'text': line} for line in lines_rest]
    
    # 写入 intermedMed_train.jsonl
    with open(train_file, 'w', encoding='utf-8') as outfile:
        for entry in data_train:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    print(f"Split completed. {len(lines_300)} lines for perplexity and {len(lines_rest)} lines for training.")

#-----------------------------------------------------------------#
# Perplexity 数据集加载
def load_intermedMed_train_dataset(train_data_file='intermedMed_train.jsonl'):
    """
    加载 intermedMed_train.jsonl 数据集。

    参数:
        train_data_file (str): intermedMed_train.jsonl 文件的路径。

    返回:
        datasets.Dataset: 加载后的训练数据集。
    """
    if not os.path.exists(train_data_file):
        print(f"File '{train_data_file}' not found.")
        return None

    dataset = load_dataset('json', data_files=train_data_file, split='train')
    return dataset

#-----------------------------------------------------------------#
# 创建并合并所有医学数据集
def create_medical_dataset():
    # 获取各个数据集的训练集和验证集
    mednli_train, mednli_val = formatted_MedNLI_dataset(num_samples=7000)
    pubmedqa_train, pubmedqa_val = formatted_PubMedQA_dataset(num_samples=7000)
    hqs_train, hqs_val = formatted_HQS_dataset(num_samples=1000)
    me_q_sum_train = formatted_ME_Q_SUM_dataset(num_samples=5000)  # 根据需要调整 num_samples
    
    # 加载 intermedMed_train.jsonl 数据集
    intermed_train = load_intermedMed_train_dataset(train_data_file='intermedMed_train.jsonl')
    
    # 合并训练集，包括 intermedMed_train 和 me_q_sum_train
    if intermed_train is not None:
        combined_train = concatenate_datasets([mednli_train, pubmedqa_train, hqs_train, me_q_sum_train, intermed_train])
    else:
        combined_train = concatenate_datasets([mednli_train, pubmedqa_train, hqs_train, me_q_sum_train])
    
    # 合并验证集
    combined_val = concatenate_datasets([mednli_val, pubmedqa_val, hqs_val])
    
    return combined_train, combined_val

#-----------------------------------------------------------------#
# 定义通用文本续写函数
def general_text_completion(model, tokenizer):
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
        generated_text = generate_summary(model, tokenizer, user_input)
        print(f"续写内容:\n{generated_text}\n")

#-----------------------------------------------------------------#
# 评估模型功能
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
            data_files="nlp_dataset_collections/HQS/HQS_test.json"
        )["train"]
        evaluate_healthquestionsum(model, tokenizer, dataset)
    elif dataset_name.lower() == 'harrison':
        evaluate_perplexity_on_harrison(model, tokenizer, masks)
    elif dataset_name.lower() == 'free':
        general_text_completion(model, tokenizer)
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

        prediction = generate_predictions(model, tokenizer, input_text)

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
            f"Premise: '{sentence1}'\n"
            f"Hypothesis: '{sentence2}'\n"
            f"Based on the premise, is the hypothesis 'entailment', 'contradiction', or 'neutral'? The answer is '"
        )

        prediction_base = generate_predictions(model, tokenizer, input_text)

        if "cont" in prediction_base:
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

def extract_message(text):
    match = re.search(r'MESSAGE:(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

def evaluate_healthquestionsum(model, tokenizer, dataset):
    print("Evaluating on HealthQuestionSum dataset...")
    references = []
    hypotheses = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i in range(len(dataset)):
        original_question = dataset[i]['CHQ']
        reference_summary = dataset[i]['Summary']

        question = extract_message(original_question)

        input_text = (
            f"A question posted by a patient is '{question}'. "
            f"The summary of the question is '"
        )

        generated_summary = generate_summary(model, tokenizer, input_text)

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

def generate_text_custom(model, tokenizer, input_ids, max_length=50):
    model.eval()
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :]

            # 使用贪心解码
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # 添加下一个标记
            generated = torch.cat((generated, next_token_id), dim=1)

            next_token = tokenizer.decode(next_token_id.squeeze())
            # 检查是否生成了句号或问号
            if next_token == '?' or next_token.strip() == '.':
                break

            # 检查结束标记
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return generated

def generate_summary(model, tokenizer, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

    generated_ids = generate_text_custom(
        model, tokenizer, input_ids, max_length=50  # 根据需要调整 max_length
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 提取摘要
    if generated_text.startswith(input_text):
        generated_summary = generated_text[len(input_text):].strip()
    else:
        generated_summary = generated_text.strip()

    # 移除引号
    if generated_summary.endswith("'"):
        generated_summary = generated_summary[:-1].strip()
    if generated_summary.startswith("'"):
        generated_summary = generated_summary[1:].strip()

    return generated_summary

def generate_predictions(model, tokenizer, input_text):
    generated_text = input_text

    model_inputs = tokenizer([generated_text], return_tensors="pt").to("cuda")

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Base model prediction
        model_output = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            return_dict=True
        )

    logits = model_output.logits
    next_token_logits = logits[:, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get next token predictions
    next_token_id = torch.argmax(probabilities, dim=-1)
    next_token = tokenizer.decode(next_token_id[0])

    return next_token 

#-----------------------------------------------------------------#
# 主函数
def main():
    # 初始化模型和标记器
    model, tokenizer, masks = initialize_model_and_tokenizer()
    
    # 创建并合并所有医学数据集
    combined_train, combined_val = create_medical_dataset()
    print(f"Combined Training Dataset Size: {len(combined_train)}")
    print(f"Combined Validation Dataset Size: {len(combined_val)}")
    
    # 支持的数据显示集列表
    supported_datasets = ['pubmedqa', 'mednli', 'hqs', 'harrison', 'free']
    
    # 循环提示用户输入数据集名称
    while True:
        dataset_name = input("Enter the dataset to evaluate (PubMedQA/MedNLI/HQS/Harrison/Free) or type 'exit' to quit: ").strip().lower()
        
        if dataset_name == 'exit':
            print("Exiting the evaluation loop.")
            break
        elif dataset_name not in supported_datasets:
            print(f"Dataset '{dataset_name}' is not supported. Please choose from {supported_datasets[:-1]} or type 'exit' to quit.")
            continue
        elif dataset_name == 'free':
            general_text_completion(model, tokenizer)
        else:
            evaluate_model_on_dataset(model, tokenizer, masks, dataset_name)
            print("-" * 50)  # 分隔线，便于阅读输出

#-----------------------------------------------------------------#
# 调用主函数
if __name__ == "__main__":
    main()