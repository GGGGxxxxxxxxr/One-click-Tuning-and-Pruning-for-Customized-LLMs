import argparse
import os
import copy
import builtins
import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# original DP is swapped into DDP for efficient and more organzied training logic
from torch.utils.data import DataLoader, DistributedSampler
import re
import pandas as pd 
import random

# llm-related library import
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets
from peft import LoftQConfig, LoraConfig, get_peft_model
from util_llm import count_llm_p_structures
from util_llm import pruning_ratio_contribution
from hypernet_llm import LLM_HyperStructure
from train_llm import llm_sp_train_one_epoch
# mask_infused_custom_llm
from custom_llms.qwen2 import Qwen2ForCausalLM
from alignment_function_llm import Group_Lasso_regularization

# ** MedicalDataset Collection ** #
#-----------------------------------------------------------------#
# MEDNLI
# ** 构建具有指定文本模板和采样的 MedNLI 数据集
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
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# HealthQuestionSum 数据集
def extract_question(text):
    """
    从文本中提取问题，处理不同的格式。
    如果文本包含 'SUBJECT:' 和 'MESSAGE:'，则提取 'MESSAGE:' 后的内容。
    否则，将整个文本视为问题。
    """
    # 检查文本是否包含 'SUBJECT:' 或 'MESSAGE:'
    # 1. 检查是否以 'Q. ' 开头
    if text.startswith('Q. '):
        question = text[3:].strip()
        return question
    
    if 'SUBJECT:' in text or 'MESSAGE:' in text:
        # 使用正则表达式提取 'MESSAGE:' 后的内容
        match = re.search(r'MESSAGE:\s*(.*)', text, re.DOTALL)
        if match:
            question = match.group(1).strip()
        else:
            # 如果无法提取，返回原始文本
            question = text.strip()
    else:
        # 将整个文本视为问题
        question = text.strip()
    return question

def format_hqs_example(example):
    # 提取问题
    question = extract_question(example['CHQ'])
    summary =  extract_question(example['Summary'])
    
    # 根据模板格式化文本
    formatted_text = (
        f"A question posted by a patient is '{question}'. "
        f"The summary of the question is '{summary}'."
    )
    
    # 返回包含新字段的字典
    return {'text': formatted_text}

def formatted_HQS_dataset(num_samples=None):
    # 加载数据集并移除不需要的列
    #training_dataset   = load_dataset("bigbio/meqsum", "meqsum_source")["train"].remove_columns(["File"])
    training_dataset   = load_dataset("json", data_files="nlp_dataset_collections/HQS/HQS_train.jsonl")['train']
    validation_dataset = load_dataset("json", data_files="nlp_dataset_collections/HQS/HQS_test.jsonl")['train'].remove_columns("q_id")

    ## more data pieces for HealthQuestionSum
    extra_train_1 = load_dataset("lighteval/me_q_sum")
    extra_train_1 = concatenate_datasets([extra_train_1['train'], extra_train_1['validation'], extra_train_1['test']])
    extra_train_1 = extra_train_1.rename_columns({
        'query': 'CHQ',
        'answer': 'Summary'
    })

    extra_train_2 = load_dataset("ruslanmv/ai-medical-chatbot")["train"].remove_columns(["Doctor"])
    extra_train_2 = extra_train_2.rename_columns({
        'Description': 'Summary',
        'Patient': 'CHQ'
    })
    training_dataset = concatenate_datasets([training_dataset, extra_train_1, extra_train_2])

    # 如果指定了 num_samples，选择前 num_samples 条数据
    if num_samples is not None:
        num_samples = min(num_samples, len(training_dataset))
        training_dataset = training_dataset.select(range(num_samples))
    
    # 应用格式化函数
    training_dataset = training_dataset.map(format_hqs_example).remove_columns(["CHQ","Summary"])
    validation_dataset = validation_dataset.map(format_hqs_example).remove_columns(["CHQ","Summary"])


    return training_dataset, validation_dataset
#-----------------------------------------------------------------#


#-----------------------------------------------------------------#
# PubMedQA
# ** 构建具有指定文本模板和采样的 PubMedQA 数据集
def format_pubmedqa_example(example):
    # 提取必要字段
    context = example['context']['contexts']
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

def format_pubmedqa_example_raw(example):
    # 提取必要字段
    context = example['CONTEXTS']
    question = example['QUESTION']
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
    # 从 Hugging Face 加载医学领域的集合
    # 修正数据集名称拼写错误，并分别加载训练集和验证集
    raw_training_path = 'nlp_dataset_collections/PubMedQA/pubMedQA_train.jsonl'
    raw_training_dataset = load_dataset("json", data_files=raw_training_path, split='train')
    raw_training_dataset = raw_training_dataset.remove_columns(
        [col for col in raw_training_dataset.column_names if col not in ["QUESTION", "CONTEXTS", "final_decision"]])

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
    
    raw_training_dataset = raw_training_dataset.map(
        format_pubmedqa_example_raw,
        remove_columns=["QUESTION", "CONTEXTS", "final_decision"]
    )

    training_dataset = concatenate_datasets([training_dataset, raw_training_dataset])

    return training_dataset, validation_dataset

def formatted_intermedMed_dataset(num_samples=None):
    train_data_file='nlp_dataset_collections/internalMed/internalMed_train.jsonl'
    val_data_file='nlp_dataset_collections/internalMed/internalMed_test.jsonl'
    train_dataset = load_dataset('json', data_files=train_data_file, split='train')
    val_dataset = load_dataset('json', data_files=val_data_file, split='train')
    if num_samples is not None:
        num_samples = min(num_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(num_samples))

    return train_dataset, val_dataset

#-------------------- 合并数据集 --------------------#
def create_medical_dataset():
    # 获取各个数据集的训练集和验证集
    mednli_train, mednli_val = formatted_MedNLI_dataset(num_samples=7000)
    pubmedqa_train, pubmedqa_val = formatted_PubMedQA_dataset(num_samples=6500)
    hqs_train, hqs_val = formatted_HQS_dataset(num_samples=1000)
    inter_train, inter_val = formatted_intermedMed_dataset(num_samples=0)
    # 合并训练集
    combined_train = concatenate_datasets([mednli_train, pubmedqa_train, hqs_train, inter_train])
    # 合并验证集
    combined_val = concatenate_datasets([mednli_val, pubmedqa_val, hqs_val, inter_val])
    
    assert len(combined_train) == 15000, f"Combined train dataset size mismatch: {len(combined_train)} != 15000"
    return combined_train, combined_val
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# *************************************************************** #
# LEGAL DOMAIN DATASET #
#-----------------------------------------------------------------#
def format_casehold_example(example):
    # 提取输入数据中的内容
    citing_prompt = example['citing_prompt']
    holding_statements = [
        example.get(f'holding_{i}', '') for i in range(5)
    ]
    label = example['label']
    
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
    formatted_text = (
        f"A citing text consisting of the context and legal citation text is '{citing_prompt}'. "
        f"Holding statement 0 is '{holding_statements[0]}', "
        f"holding statement 1 is '{holding_statements[1]}', "
        f"holding statement 2 is '{holding_statements[2]}', "
        f"holding statement 3 is '{holding_statements[3]}', "
        f"and holding statement 4 is '{holding_statements[4]}'. "
        f"Choose the correct corresponding holding statement. "
        f"The correct answer is holding statement {label}, which is the {idx} statement."
    )

    return {'text': formatted_text}

def format_casehold_example_qa(example):
    citing_prompt = example['citing_prompt']
    holding_statements = [example.get(f'holding_{i}', '') for i in range(5)]
    label = example['label']
    
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
    
    # Construct the input text
    formatted_text = (
        f"A citing text consisting of the context and legal citation text is '{citing_prompt}'. "
        f"Holding statement 0 is '{holding_statements[0]}', "
        f"holding statement 1 is '{holding_statements[1]}', "
        f"holding statement 2 is '{holding_statements[2]}', "
        f"holding statement 3 is '{holding_statements[3]}', "
        f"and holding statement 4 is '{holding_statements[4]}'. "
        f"Choose the correct corresponding holding statement. "
    )
    
    # The correct answer
    answer_text = f"The correct answer is holding statement {label}, which is the {idx} statement."

    # Return both the input and the answer
    return {'text': formatted_text, 'answer': answer_text}


def formatted_casehold_dataset(num_samples=None, args=None):
    raw_train_2000 = load_dataset('json', data_files="nlp_dataset_collections/CaseHold/casehold_train_clean_2000.jsonl", split='train')
    
    # 应用格式化函数到模板数据集并获取最大长度
    formatted_raw_train_2000 = raw_train_2000.map(format_casehold_example)
    max_length = max(len(example['text']) for example in formatted_raw_train_2000)
    print(f"Maximum formatted string length in CaseHold raw_train_2000: {max_length}")

    # 加载 CaseHold 数据集
    ds = load_dataset("casehold/casehold", "all")['train']
    ds_val = load_dataset("casehold/casehold", "all")['validation']

    # 应用格式化函数并过滤训练集，保留格式化后文本长度不超过 max_length 的样本
    if args!=None and args.loss_on_answer == False:
        filtered_train_dataset = ds.map(format_casehold_example).filter(lambda x: len(x['text']) <= max_length)
        filtered_val_dataset   = ds_val.map(format_casehold_example).filter(lambda x: len(x['text']) <= max_length)
    else:
        filtered_train_dataset = ds.map(format_casehold_example_qa).filter(lambda x: (len(x['text']) + len(x['answer'])) <= max_length)
        filtered_val_dataset   = ds_val.map(format_casehold_example_qa).filter(lambda x: (len(x['text']) + len(x['answer'])) <= max_length)

    # 如果指定了 num_samples，选择前 num_samples 条数据
    if num_samples is not None:
        train_dataset = filtered_train_dataset.select(range(min(num_samples, len(ds))))
    
    val_dataset = filtered_val_dataset.select(range(min(500, len(ds_val))))

    # 移除不需要的列
    train_dataset = train_dataset.remove_columns(
        ['citing_prompt', 'holding_0', 'holding_1', 'holding_2', 'holding_3', 'holding_4', 'label', 'example_id']
    )
    val_dataset   = val_dataset.remove_columns(
        ['citing_prompt', 'holding_0', 'holding_1', 'holding_2', 'holding_3', 'holding_4', 'label', 'example_id']
    )

    return train_dataset, val_dataset


def format_billsum_example(example):
    # 使用模板格式化文本
    formatted_text = (
        f"A bill text is '{example['source']}'. "
        f"The summary of the bill is '{example['summary']}'."
    )
    return {'text': formatted_text}

def format_billsum_example_qa(example):
    # 使用模板格式化文本
    formatted_text = (
        f"A bill text is '{example['source']}'. "
        f"Please summary this bill."
    )
    answer = (
        f"The summary of the bill is '{example['summary']}'."
    )

    return {'text': formatted_text, 'answer': answer}


def formatted_billsum_dataset(num_samples=None, args=None):
    ds = load_dataset("json", data_files="nlp_dataset_collections/BillSum/billsum_train_2000.jsonl")['train']
    ds_val = load_dataset("json", data_files="nlp_dataset_collections/BillSum/billsum_test_200.jsonl")['train']

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    
    if args!=None and args.loss_on_answer == False:
        # apply string formatted-function
        train_dataset = ds.map(format_billsum_example).remove_columns(
            ['source', 'summary']
        )
        val_dataset   = ds_val.map(format_billsum_example).remove_columns(
            ['source', 'summary']
        )
    else:
        train_dataset = ds.map(format_billsum_example_qa).remove_columns(
            ['source', 'summary']
        )
        val_dataset   = ds_val.map(format_billsum_example_qa).remove_columns(
            ['source', 'summary']
        )

    return train_dataset, val_dataset


def formatted_multilegalpile_dataset(num_samples=None):
    ds = load_dataset("json", data_files='nlp_dataset_collections/MultiLegalPile/multilegalpile_300.jsonl')['train']
    val_dataset = ds.remove_columns(["language", 'type', 'jurisdiction'])

    val_dataset = val_dataset.map(lambda example: {'answer': ''})
    
    # If num_samples is specified, limit the dataset size
    if num_samples is not None:
        val_dataset = val_dataset.select(range(min(num_samples, len(ds))))
    
    return val_dataset

def create_legal_dataset(args):
    # 加载数据集
    billsum_train, billsum_val = formatted_billsum_dataset(num_samples=2000, args=args)
    casehold_train, casehold_val = formatted_casehold_dataset(num_samples=13000, args=args)
    perplexity_val = formatted_multilegalpile_dataset()

    # 合并训练集和验证集
    combined_train = concatenate_datasets([billsum_train, casehold_train])
    combined_val = concatenate_datasets([billsum_val, casehold_val, perplexity_val])
    
    # 确保训练集大小正确
    assert len(combined_train) == 15000, f"Combined train dataset size mismatch: {len(combined_train)} != 15000"

    # 从训练集中随机抽取 2000 条样本，并将其添加到验证集中
    random_sample_indices = random.sample(range(len(combined_train)), 2000)
    train_samples_for_val = combined_train.select(random_sample_indices)
    
    # 将抽取的样本添加到验证集中
    combined_val = concatenate_datasets([combined_val, train_samples_for_val])

    # 检查数据集大小
    assert len(combined_val) == (len(billsum_val) + len(casehold_val) + len(perplexity_val) + 2000), \
        f"Combined val dataset size mismatch after sampling: {len(combined_val)} != expected size"

    return combined_train, combined_val

    


#-----------------------------------------------------------------#
# WIKITEXT
# ** build WIKITEXT Dataset (text completion)
# ** empty string filtered in order to improve training quality
def filter_empty_text(example):
    # Filter out empty text examples
    return example['text'] != ''

def formatted_wikitext_dataset():
    # Load the Wikitext-103 dataset (train and validation splits)
    nlp_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split='train')
    val_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split='validation')
    
    # Print the dataset info and the first sample
    print("=====> Wikitext Dataset Config & First Sample: <=====\n")
    print(nlp_dataset)
    
    # Confirm the dataset length before filtering
    print(f"Training size: {len(nlp_dataset)} samples")
    print(f"Validation size: {len(val_dataset)} samples")
    
    # Filter out empty text entries
    nlp_dataset = nlp_dataset.filter(filter_empty_text)
    val_dataset = val_dataset.filter(filter_empty_text)
    
    # Confirm the dataset length after filtering
    print(f"Filtered Training size: {len(nlp_dataset)} samples")
    print(f"Filtered Validation size: {len(val_dataset)} samples")
    
    return nlp_dataset, val_dataset
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# ** build InternalMed_Harrison_text dataset
# ** it is a medicial textbook, the corresponding data is extracted from a MedicalDomain collections
def formatted_InterMed_dataset():
    # load the medical-domain collections from huggingface source
    general_dataset = load_dataset('cogbuji/medqa_corpus_en', 'core_clinical')['train']
    # filter the InternalMed_Harrison from the general collections
    dataset = general_dataset.filter(lambda example: example['source'] == 'textbooks/en/InternalMed_Harrison.txt').remove_columns(["source"])
    split_dataset = dataset.train_test_split(test_size=0.15, shuffle=True, seed=42)
    training_dataset = split_dataset['train']
    print(training_dataset[0])   
    validation_dataset = split_dataset['test']

    return training_dataset, validation_dataset
#-----------------------------------------------------------------#






#-----------------------------------------------------------------#
def format_agnews_example(example):
    # Extract necessary fields
    sentence = example['text']
    class_label = example['label']
    
    # Format the text based on the provided template
    formatted_text = (
        f"Predict the #class_label# from '0', '1', '2' or '3' based on the content of #sentence#. "
        f"#sentence#: '{sentence}'. Predicted #class_label#: {class_label}"
    )
    
    # Return the new dictionary with formatted text
    return {'text': formatted_text}

# ** build AG NEWs dataset
# ** it is a Classification dataset for News classification on Business, Science, Sports ...
# ** a customized template is formulated for LLM to do the expected prediction behavior, similar to MedNLI
def formatted_AGNews_dataset():
    train_dataset      = load_dataset("fancyzhx/ag_news")["train"]
    validation_dataset = load_dataset("fancyzhx/ag_news")["test"]
    # apply the template for dataset mapping
    train_dataset      = train_dataset.map(format_agnews_example)
    validation_dataset = validation_dataset.map(format_agnews_example)

    return train_dataset, validation_dataset
#-----------------------------------------------------------------#


'''
legal_train, legal_val = create_legal_dataset()
text_lengths = [len(example['text']) for example in legal_train]

# 转换为 Pandas Series 并查看描述统计信息
text_length_series = pd.Series(text_lengths)
print(text_length_series.describe())
'''
'''
legal_train,legal_val = formatted_billsum_dataset(2000, args=None)
legal_train_1,legal_val_2 = formatted_casehold_dataset(13000, args=None)
train = concatenate_datasets([legal_train_1, legal_train])
print(legal_train[1])
'''




