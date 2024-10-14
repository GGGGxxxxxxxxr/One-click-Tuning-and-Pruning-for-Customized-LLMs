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

# llm-related library import
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoftQConfig, LoraConfig, get_peft_model
from util_llm import count_llm_p_structures
from util_llm import pruning_ratio_contribution
from hypernet_llm import LLM_HyperStructure
from train_llm import llm_sp_train_one_epoch
# mask_infused_custom_llm
from custom_llms.qwen2 import Qwen2ForCausalLM
from alignment_function_llm import Group_Lasso_regularization


#-----------------------------------------------------------------#
# MEDNLI
# ** build MedNLI Dataset (medical content Classification)
# ** data reordered with pre-fixed text template to make it more friendly to LLM understanding
def format_example(example):
    # Extract necessary fields
    sentence1 = example['sentence1']
    sentence2 = example['sentence2']
    gold_label = example['gold_label']
    
    # Format the text based on the provided template
    formatted_text = (
        f"Predict the #gold_label# from 'entailment', 'contradiction' or 'neutral' based on the content of #sentence1# and #sentence2#. "
        f"#sentence1#: '{sentence1}', #sentence2#: '{sentence2}'. Predicted #gold_label#: {gold_label}"
    )
    
    # Return the new dictionary with formatted text
    return {'text': formatted_text}

def formatted_MedNLI_dataset(train_data_file='/home/user1/workspace/leilu/AutoTrainOnce/nlp_dataset_collections/medNLI/mli_train_v1.jsonl', val_data_file='/home/user1/workspace/leilu/AutoTrainOnce/nlp_dataset_collections/medNLI/mli_dev_v1.jsonl'):
    # Load the dataset and remove unnecessary columns
    train_set = load_dataset("json", data_files=train_data_file).remove_columns(
        ["pairID", "sentence1_parse", "sentence1_binary_parse", "sentence2_parse", "sentence2_binary_parse"]
    )
    val_set   = load_dataset("json", data_files=val_data_file).remove_columns(
        ["pairID", "sentence1_parse", "sentence1_binary_parse", "sentence2_parse", "sentence2_binary_parse"]
    )
    # Apply the formatting function and remove original columns
    train_set = train_set.map(format_example).remove_columns(["sentence1", "sentence2", "gold_label"])
    val_set   = val_set.map(format_example).remove_columns(["sentence1", "sentence2", "gold_label"])
    return train_set['train'], val_set['train']
#-----------------------------------------------------------------#


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
# ** build PubMedQA dataset
# ** it is a Research Question in MedicialDomain where:
# ** an ##answer## [Yes/NO] is decided given the ##Question## and ##context## (supporting details)
# ** version0: reserve "question" "context" (several subattributes within) "final_decision" (predictable answer)
def formatted_PubMedQA_dataset():
    # load the medical-domain collections from huggingface source
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")['train'].remove_columns(["pubid","long_answer"])

    # split into training_set for llm tuning & validation_set for hypernet() mask generation
    split_dataset = dataset.train_test_split(test_size=0.15, shuffle=True, seed=42)
    training_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']

    return training_dataset, validation_dataset
#-----------------------------------------------------------------#



datase = formatted_PubMedQA_dataset()