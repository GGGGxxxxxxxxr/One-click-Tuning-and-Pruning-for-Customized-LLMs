# official implementation for "All-in-One Tuning and Structural Pruning for Domain-specific LLMs"
# https://arxiv.org/abs/2412.14426

# version2.0 main updates:
# 1) more choices for pruning space
# 2) reimplemented group lasso approximation solution, subsitutional from direct grouplasso loss
# 3) serve for more general purpose deployment

# waiting for updates:
# 1) final compressed model inference:
#    - trition kernel for index selection addition
#    - ATP model --> final compressed model conversion
# 2) QLoRA for ATP's upscaling towards 13B or even 30B

# to run this script:
# CUDA_VISIBLE_DEVICES=0,1,2, ..  torchrun --nproc_per_node=n main_llm_atp_disp.py --model llama2-7b --tuning-method lora --pruning-method DISP --lr 1e-4 --lora-rank 32 --epochs 2 --control-epochs 1 --svd-init

## model quantization support added

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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig

# Initialize FSDP with mixed precision
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from functools import partial

# 8bit optimizer for memory efficent training
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, replace_lora_weights_loftq

# llm-related library import
from transformers import AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq
from datasets import load_dataset
from util_llm import count_llm_p_structures, count_total_params, count_trainable_parameters, count_total_prunable_params
from util_llm import customized_lora_substitution
from hypernet_llm import LLM_HyperStructure
from train_llm_disp import llm_sp_train_one_epoch
from build_dataset import formatted_MedNLI_dataset, formatted_wikitext_dataset, formatted_AGNews_dataset, create_medical_dataset, create_legal_dataset, formatted_alpaca_dataset, formatted_alpaca_gpt_dataset
# mask_infused_custom_llm
from custom_llms.qwen2 import Qwen2ForCausalLM
from custom_llms.llama_disp import LlamaForCausalLM
from alignment_function_llm import Group_Lasso_regularization_DISP
from custom_llms.llama import LlamaDecoderLayer

parser = argparse.ArgumentParser(description='PyTorch Implementation for ATO on LLM LoRA & Structure Pruning')
#-----------------------------------------------------------------#
# training-related args
parser.add_argument('--model', metavar='MODEL', default='qwen-0.5b',
                    help='name of the target llm model')
parser.add_argument('--use-8bit-training', dest='use_8bit_training',action='store_true',
                    help='use 8bit AdamW optimizer to save GPU mem for training')
parser.add_argument('--num-gpus', default=8, type=int, metavar='G',
                    help='specify how many GPUs for launching the training workload')
parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bsz', default=4, type=int,metavar='N',
                    help='training batchsize of a single process, DDP is utilized instead of DP in ATO implementation')
parser.add_argument('--lr', metavar='LR', default=1e-4, type=float,
                    help='basic initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-decay', default=1e-4, type=float,metavar='W', 
                    help='weight decay (default: 1e-4)')
parser.add_argument('--lmd', default=10, type=float, metavar='W', 
                    help='group lasso lamd (default: 10)')
parser.add_argument('--epsilon', default=0.1, type=float, metavar='M',
                    help='epsilon in OTO')
parser.add_argument('--print-freq', default=100, type=int, metavar='N', 
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--log-loss', action='store_true',
                    help='Enable logging of the training loss per step')
parser.add_argument('--log-interval', default=20, type=int,
                    help='training log intervals of ATP steps')
parser.add_argument('--valid-interval', default=40, type=int,
                    help='validation intervals on the fixed pruning decisions')
parser.add_argument('--gltrack-interval', default=20, type=int,
                    help='group lasso loss tracking intervals')
parser.add_argument('--loss-on-answer', action='store_true', 
                    help='If set, calculate loss on the entire input including the context')
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# methdology-related args
parser.add_argument('--dataset', default='MedNLI', type=str,
                    help='specify the domain-specifc dataset for LLM pararm tuning, **build-in ready2use selections: wikitext, MedNLI, AGNews, legal, medical, alpaca')
parser.add_argument('--pruning-method', default='DISP', type=str,
                    help='head-wise MHA pruning **[head_wise], layer-aware-uniform head pruning **[layer_uniform_attn], more fine-grained within attention-head pruning **[inner], or stronger pruning space **[DISP]')
parser.add_argument('--tuning-method',  default='lora', type=str,
                    help='lora tuning **[lora] or full param tuning **[full]')
parser.add_argument('--pruning-ratio-target', default=0.5, type=float,
                    help='Pruning Rate')
parser.add_argument('--start-epoch-control', default=0, type=int,
                    help='which epoch to start the training of controller_network')
parser.add_argument('--control-epochs', default=4, type=int,
                    help='how many epochs for controller_network_training')
parser.add_argument('--control-step', default=1, type=int,
                    help='HyperNet() param update gap, default = 1')
parser.add_argument('--start-epoch-regularization', default=0, type=int,
                    help='which epoch to start the loss of group-sparsity-related constratins')
parser.add_argument('--svd-init', action='store_true', 
                    help='whether to use SVD initialization for LoRA layers')
parser.add_argument('--lora-rank', default=32, type=int,
                    help='rank of the low-rank matrices in LoRA layers')
parser.add_argument('--quantization', action='store_true', help="Enable weight quantization for ATP upscaling")
parser.add_argument('--loftq-init', action='store_true', help="Enable LoftQ SVD init for lora weights")
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument("--local-rank", type=int, default=0)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# controller_network for mask generation args 
# (followed the ATO's implementation, further revision here)
parser.add_argument('--base',     default=3.0, type=float)
parser.add_argument('--base_p',   default=1.0, type=float)
parser.add_argument('--grad-mul', default=1.0, type=float)
parser.add_argument('--gl-lam',   default=1.0, type=float)
parser.add_argument('--num-key-values',  default=2, type=int)
#-----------------------------------------------------------------#

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        if is_master:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

#-----------------------------------------------------------------#
# checkpoint saving for each epoch
def save_checkpoint(
    epoch, 
    cur_mask_vec,
    model=None, 
    filename="/orange/sgao1/atp_llm_dir/llm_atpdisp_lora_qa_quantize.pth.tar"
):
    """
    Save the training checkpoint including model, hyper_net weights, optimizers, and current mask vector.

    Args:
    epoch (int): Current epoch number.
    model (torch.nn.Module, optional): The main model (target_llm).
    hyper_net (torch.nn.Module, optional): The hyper network.
    optimizer_llm (torch.optim.Optimizer, optional): Optimizer for the main model.
    optimizer_hyper (torch.optim.Optimizer, optional): Optimizer for the hyper network.
    cur_mask_vec (torch.Tensor, optional): Current mask vector.
    filename (str): Path to save the checkpoint file.
    """
    # Initialize the state dictionary
    state = {'epoch': epoch,
             'mask_vec': cur_mask_vec}

    # Store state_dicts only if the corresponding component is not None
    if model is not None:
        state['model_state_dict'] = (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        )

    # Save only on the main process to avoid multiple processes writing the file
    if dist.get_rank() == 0:
        torch.save(state, filename)
        print(f"Checkpoint saved at epoch {epoch} to {filename}\n")

    # Synchronize all processes to ensure consistency
    dist.barrier()


def save_fsdp_checkpoint(epoch, model, cur_mask_vec, filename="/orange/yonghui.wu/sgao1/llm_pruning_test.pth.tar"):
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
            state = {
                'model_state_dict': cpu_state,
                'mask_vec': cur_mask_vec,
            }

            # Save only on the main process to avoid multiple processes writing the file
            if torch.distributed.get_rank() == 0:
                torch.save(state, filename)
                print(f"Checkpoint saved at epoch {epoch} to {filename}\n")
    
    dist.barrier()
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def load_checkpoint(model, hyper_net, optimizer_llm, optimizer_hyper, filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))  # adjust map_location as needed
    model.load_state_dict(checkpoint['model_state_dict'])
    #hyper_net.load_state_dict(checkpoint['hyper_net_state_dict'])
    #optimizer_llm.load_state_dict(checkpoint['optimizer_llm_state_dict'])
    #optimizer_hyper.load_state_dict(checkpoint['optimizer_hyper_state_dict'])
    cur_mask_vec = checkpoint['mask_vec']
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: epoch {epoch}")
    return epoch, cur_mask_vec
#-----------------------------------------------------------------#

def training_setup():
    # Initialize the process group
    gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://')


def training_cleanup():
    dist.destroy_process_group()


def main():
    #-----------------------------------------------------------------#
    # init DistributedDataParallel
    training_setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    setup_for_distributed(rank==0)
    
    print("""
                  ██████████
              ████          ████
          ████       ATP:       ████
       ████   All-in-One Tuning     ████
       ████    & Structural Pruning ████
          ████    version2.0    ████
              ████          ████
                  ██████████
    """)

    print("\n=====> DDP Training ENV Initialization Done. <=====")
    #-----------------------------------------------------------------#
    args = parser.parse_args()
    
    #-----------------------------------------------------------------#
    # Reproducibility
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # pre-trained LLM initialization
    # Huggingface support is pulled to make sure the generalization ability of our scripts.
    # ** current support: {qwen2-0.5b, llama2-7b, llama3-8b}
    if args.tuning_method == 'lora':
        print("\n[INFO] Tuning Method: LoRA")
        print("LoRA has been enabled for tuning. Distributed Data Parallel (DDP) mode is selected.")
        init_device = device
    else:
        print("\n[INFO] Tuning Method: Full-Parameter")
        print("Full-parameter tuning has been enabled. Fully Sharded Data Parallel (FSDP) mode is selected.")
        init_device = 'cpu'

    print(f"\n[INFO]=====> Initializing pre-trained model: '{args.model}' from Huggingface <=====")
    if args.model == 'qwen-0.5b':
        # Qwen-0.5B Model Initialization
        model_cfg = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to(init_device)
        args.num_key_values = model_cfg.num_key_value_heads

        print("\n[INFO]=====> Model structure for 'qwen-0.5b': <=====")
        print(model)
        print("[INFO]=====> Model structure ends. <=====\n")

    elif args.model == 'llama2-7b':
        if args.quantization == False:
            # LLaMA 2-7B Model Initialization
            print("\n[INFO] LLaMA 2-7B detected. Initializing with API token...")
            api_token = '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'
            model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
            print(f"\n[INFO] Pretraining TP: {model_cfg.pretraining_tp}")
            
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'left'
            
            model = LlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                token=api_token
            ).to(init_device)
            model.resize_token_embeddings(len(tokenizer))
            args.num_key_values = model_cfg.num_key_value_heads
        
        else:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  
                bnb_4bit_compute_dtype=torch.bfloat16, 
                bnb_4bit_use_double_quant=False, 
                bnb_4bit_quant_type="nf4"
            )
            # LLaMA 2-7B Model Initialization
            print("\n[INFO] LLaMA 2-7B detected. Initializing with API token...")
            api_token = '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'
            model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
            print(f"\n[INFO] Pretraining TP: {model_cfg.pretraining_tp}")
            
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'left'

            model = LlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                device_map="auto",  
                quantization_config=bnb_config,
                token=api_token
            ).to(init_device)

            model.resize_token_embeddings(len(tokenizer))
            args.num_key_values = model.config.num_key_value_heads

            print("\n[INFO] LLaMA 2-7B Model initialized successfully!")
            print("\n[INFO] Pretrained Weights are quantized to save on the GPU memory consumption...")
            print(model)


    elif args.model == 'llama3-8b':
        # LLaMA 3-8B Model Initialization
        print("\n[INFO] LLaMA 3-8B detected. Initializing with API token...")
        api_token = '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'
        model_cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B", token=api_token)
        print(f"[INFO] Pretraining TP: {model_cfg.pretraining_tp}")
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=api_token)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'left'
        
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            token=api_token
        ).to(init_device)
        model.resize_token_embeddings(len(tokenizer))
        args.num_key_values = model_cfg.num_key_value_heads
        print(f"[INFO] Padding Token: {tokenizer.pad_token}")

    else:
        # Model Not Implemented
        print("\n[INFO]=====> Model not implemented yet! Exiting the system. <=====")
        sys.exit()
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # counting prunable structures for LLMs
    if args.pruning_method == 'inner':
        print("\n[INFO]=====> Fine-grained pruning is ENABLED. Pruning would be conducted within MHA. <=====")
        p_structures = count_llm_p_structures(model = model, model_config = model_cfg, pruning_scheme = args.pruning_method)
    elif args.pruning_method == 'layer_uniform_attn':
        print("\n[INFO]=====> Layer-uniform attn pruning is ENABLED. Pruning would be conducted within MHA but the layer-wise pruning pattern would be shared across all heads. <=====")
        p_structures = count_llm_p_structures(model = model, model_config = model_cfg, pruning_scheme = args.pruning_method)
    elif args.pruning_method == 'DISP':
        print("\n[INFO]=====> DISP pruning space is ENABLED. Pruning would be conducted within MHA but on the input dimension. Refer to DISP for more detailed implementation. <=====")
        p_structures = count_llm_p_structures(model = model, model_config = model_cfg, pruning_scheme = args.pruning_method)
    else:
        print("\n[INFO]=====> AttentionHead pruning is ENABLED. Pruning would be conducted head-wisely on Query. <=====")
        print("CURRENTLY NOT SUPPORTED!")
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # LoRA integration or full-param tuning
    # currently we use full-param, LoRA feature would be developed later
    if args.tuning_method == 'lora':
        if args.quantization == True:
            '''
                ** in order to fit LoRA with ATP Implementation, customized LoRALinear with mask capability is required
                ** thus instead of Peft library, we customize LoRA Infusion.
            '''
            print("=====> LoRA Tuning Initialization, pre-trained weights would be frozen during the following stages. <=====")
            lora_config = LoraConfig(
                            #init_lora_weights="loftq",
                            r=args.lora_rank,
                            lora_alpha=1,
                            target_modules=["q_proj","k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
                            lora_dropout=0.1,
                            task_type="CAUSAL_LM"
                        )
            # lora detailed configuration
            print(f"  r: {lora_config.r}")
            print(f"  lora_alpha: {lora_config.lora_alpha}")
            print(f"  target_modules: {lora_config.target_modules}")
            print(f"  lora_dropout: {lora_config.lora_dropout}")
            print(f"  bias: {lora_config.bias}")
            # fuse lora module into pre-trained target llm
            model = get_peft_model(model, lora_config)
            print("=====> LoRA infusion done. <=====\n")
            model.print_trainable_parameters()  

            # LoftQ lora initialization?
            if args.loftq_init == True:
                for _ in range(5):
                    replace_lora_weights_loftq(model)
                print(model)
        

        else:
            customized_lora_substitution(model, rank=args.lora_rank, dropout=0.1, svd_init=args.svd_init)

        if args.svd_init == True:
            print(f"\n[INFO]=====> LoRA initialization with SVD decomposition with top_r: {args.lora_rank} <=====")
        else:
            print(f"\n[INFO]=====> LoRA initialization with common solution with rank: {args.lora_rank}. <=====")

        print("\n[INFO]=====> LoRA infusion done. <=====")

    else:
        print("\n[INFO]=====> Full-param tuning Initialization Done. Pre-trained weights would be tuned during the following stages. <=====")
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # build controllernetwork for mask generation
    print("\n[INFO]=====> Initialize Pruning_Decision_Generator(PDG) based on [prunable_structure, temperature, base]. <=====")
    hyper_net = LLM_HyperStructure(p_structure = p_structures, T = 0.4, base = 3, args = args).to(dtype=torch.bfloat16).to(device)
    cur_maskVec = hyper_net(dummy=0)
    cur_mask_init = hyper_net.transform_output(cur_maskVec)
    '''
    DEBUGGING:
    initialized maskVec
    '''
    number_of_zeros = (cur_maskVec == 0).sum().item()
    print(f"\n[DEBUG]: initialized binary mask has {number_of_zeros} masked dimension within the vector.")

    print("\n[DEBUG]random initialized Mask:\n", cur_maskVec, cur_maskVec.size())
    print("\n[DEBUG]random initialized Mask for LLM Inference:\n", cur_mask_init)
    print("\n[INFO]=====> Pruning_Decision_Generator Initialization Done. <=====")
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # user-specified dataset selection & initialization
    torch.backends.cuda.enable_flash_sdp(True)
    ## general purpose dataset
    if args.dataset   == "wikitext":
        nlp_dataset, val_dataset = formatted_wikitext_dataset()
    
    elif args.dataset == 'AGNews':
        nlp_dataset, val_dataset = formatted_AGNews_dataset()
    
    # update: for alpaca, we are right now testing which loss modeling would be better
    elif args.dataset == 'alpaca':
        nlp_dataset, val_dataset = formatted_alpaca_dataset(args=args, num_val_samples=10000)
        #assert args.loss_on_answer == True, "If Alpaca dataset is used, then the model loss is computed on [answer] only."
    elif args.dataset == 'alpaca_gpt':
        nlp_dataset, val_dataset = formatted_alpaca_gpt_dataset(args=args, num_val_samples=5000)
    ## domain-specific dataset
    elif args.dataset == 'MedNLI':
        nlp_dataset, val_dataset = formatted_MedNLI_dataset()
    elif args.dataset == 'medical':
        nlp_dataset, val_dataset = create_medical_dataset(args=args)
        print(val_dataset)
    elif args.dataset == 'legal':
        nlp_dataset, val_dataset = create_legal_dataset(args=args)

    print(f"\n[INFO] Dataset Selection: {args.dataset}")
    print("\n[INFO]=====> templated dataset config: <=====\n")
    print(nlp_dataset)
    print(val_dataset)

    # Print the 85th samples just for dataset cleaning verfication
    print("\n[INFO]=====> sequence samples: <=====")
    print(nlp_dataset[85])  
    print(val_dataset[85])

    # tokenize the NLP dataset
    # ** please determine whether to model loss on 'answer' or 'entire sentences'
    def tokenize_function(examples):
        # Tokenize with truncation enabled but without padding
        inputs  = tokenizer(examples["text"], padding=False)

        if not args.loss_on_answer:
            if examples["answer"]:
                answers = tokenizer(examples["answer"], padding=False)
                # Add the EOS token ID at the end of each tokenized input
                eos_token_id = tokenizer.eos_token_id 
                if eos_token_id is None:
                    raise ValueError("Your tokenizer does not have an eos_token_id. Please set an EOS token for your tokenizer.")
                input_ids = inputs["input_ids"] + answers["input_ids"] + [eos_token_id]
                attention_mask = inputs["attention_mask"] + answers["attention_mask"] + [1]
                labels = input_ids
            else:
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = input_ids
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
    
        else:
            # Ensure 'answer' key is present, otherwise handle it gracefully
            if 'answer' not in examples:
                raise ValueError("The 'answer' key is missing in the dataset but 'loss_on_answer' is set to True.")
        
            if examples["answer"]:
                answers = tokenizer(examples["answer"], padding=False)
                # Add the EOS token ID at the end of each tokenized input
                eos_token_id = tokenizer.eos_token_id 
                if eos_token_id is None:
                    raise ValueError("Your tokenizer does not have an eos_token_id. Please set an EOS token for your tokenizer.")
                input_ids = inputs["input_ids"] + answers["input_ids"] + [eos_token_id]
                attention_mask = inputs["attention_mask"] + answers["attention_mask"] + [1]
                labels = [-100] * len(inputs["input_ids"]) + answers["input_ids"] + [eos_token_id]
            else:
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = input_ids
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
    
    tokenized_datasets = nlp_dataset.map(tokenize_function).remove_columns(["text", 'answer'])
    tokenized_valsets  = val_dataset.map(tokenize_function).remove_columns(["text", 'answer'])
    print("\n[INFO]=====> tokenized dataset config: <=====")
    print(tokenized_datasets)
    print(tokenized_valsets)
    print("\n[INFO]=====> tokenized samples: <=====")
    print(tokenized_datasets[85])
    print(tokenized_valsets[85])
    print("\n[INFO]=====> NLP Datasets Initialization Done. <=====")


    # config training dataloader
    data_collator  = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    ddp_sampler    = DistributedSampler(tokenized_datasets, num_replicas=world_size, rank=rank)
    ddp_sampler1   = DistributedSampler(tokenized_valsets, num_replicas=world_size, rank=rank)
    nlp_dataloader = DataLoader(
                        tokenized_datasets,
                        batch_size=args.bsz,
                        sampler=ddp_sampler,
                        collate_fn=data_collator)
    val_dataloader = DataLoader(
                        tokenized_valsets,
                        batch_size=args.bsz,
                        sampler=ddp_sampler1,
                        collate_fn=data_collator)
    print("\n[INFO]=====> NLP DDP_DataLoader Initialization Done. <=====")
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # Training Optimizier (llm target loss is automatically calculated within the CasualLM.forward() with CrossEntropyLoss(), we only need to take care of the GroupLasso and HyperNet() update)
    # 1. For both target LLM and HyperNet(), AdamW() is applied as the param optimizer
    # 2. For HyperNet() and target LLM, common CosLR is applied for learningrate adjusting for smooth LR dropping
    # Pre-training from scratch probably needs CosWarmRestart() but as we have a inital point from pre-trained, we could go for CosLR directly.

    # a) optimizer for Mask HyperNet()
    hyper_net_ddp   = DDP(hyper_net, device_ids=[device])
    hyper_params    = hyper_net_ddp.parameters()
    if args.use_8bit_training == True:
        optimizer_hyper = bnb.optim.AdamW8bit(hyper_params,lr  = 1e-4)
    else:
        optimizer_hyper = torch.optim.AdamW(hyper_params,  lr  = 3e-4)
    
    print("\n[INFO]=====> Trainable parameters for Pruning_Decision_Generator: <=====")
    for name, param in hyper_net_ddp.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")

    # b) optimzer for target LLM
    if args.tuning_method != 'lora':
        llm_ddp = FSDP(model, device_id=device, auto_wrap_policy=llama_auto_wrap_policy, use_orig_params=False, mixed_precision=mixed_precision_policy)
        print("[INFO]FSDP wrapper with mixed precision has been enabled.")
    else:
        llm_ddp = DDP(model, device_ids=[device])
        print("[INFO]DDP wrapper has been enabled.")

    #llm_ddp         = torch.compile(llm_ddp)
    if args.use_8bit_training == True:
        optimizer_llm   = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, llm_ddp.parameters()),lr = args.lr)
    else:
        optimizer_llm   = torch.optim.AdamW(filter(lambda p: p.requires_grad, llm_ddp.parameters()),  lr = args.lr)

    # learning rate scheduler
    scheduler_llm   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_llm, T_max=args.epochs, eta_min=1e-4)
    scheduler_hyper = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hyper, T_max=args.control_epochs, eta_min=1e-4)

    print("\n[INFO]=====> Trainable parameters for target_LLM: <=====")
    print(llm_ddp)
    for name, param in llm_ddp.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")

    print("\n[INFO]=====> Training Optimizers and Schedulers Initialization Done. <=====")
    #-----------------------------------------------------------------#
    if args.tuning_method != "lora":
        print("[INFO]ShardedGradScaler has been intialized for FSDP.")
        scaler = ShardedGradScaler()

    #-----------------------------------------------------------------#
    # group_lasso_loss module intialization
    grouplasso_module = Group_Lasso_regularization_DISP(args = args, target_llm_cfg = model_cfg, prunable_structure = p_structures).to(device=init_device)
    print("\n[INFO]=====> Group_Lasso Sparsity Module Initialization Done. <=====")
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # compute pruning contribution of individual mask indicator
    #pruning_contribution = pruning_ratio_contribution(model_cfg=model_cfg)
    #print("=====> Pruning_Contribution: <=====\n")
    #print(pruning_contribution)
    total_params            = count_total_params(model)
    total_prunable_params   = count_total_prunable_params(model_name=args.model)
    total_trainable_params  = count_trainable_parameters(model)

    print(f"\n[INFO]LLM total params: {total_params - total_trainable_params}, total_prunable_params: {total_prunable_params}, total trainable params:{total_trainable_params}, ratio = {total_trainable_params / (total_params-total_trainable_params)}")
    #-----------------------------------------------------------------#
    
    #-----------------------------------------------------------------#
    # Training Process
    print("[PROCESS]=====> Begin Training: <=====\n")
    if args.resume != None:
        print(f"[PROCESS]=====> Resume Training from {args.resume}: <=====\n")
        epoch, cur_maskVec = load_checkpoint(model=llm_ddp.module, hyper_net=hyper_net_ddp.module, optimizer_llm=optimizer_llm, optimizer_hyper=optimizer_hyper, filename=args.resume)
        start_epoch = epoch + 1
    else:
        print(f"[PROCESS]=====> New Training Progress Launched. <=====\n")
        start_epoch = args.start_epoch

    for epoch in range(start_epoch, args.epochs):
        # data shuffle epoch-wisely
        ddp_sampler.set_epoch(epoch)
        ddp_sampler1.set_epoch(epoch)

        # train for one epoch
        cur_maskVec, loss_log = llm_sp_train_one_epoch(nlp_dataloader=nlp_dataloader, nlp_hypernet_dataloader=val_dataloader, target_llm=llm_ddp, 
                                            hyper_net=hyper_net_ddp , optimizer_llm=optimizer_llm, optimizer_hyper=optimizer_hyper, epoch=epoch, total_epochs=args.epochs, cur_mask_vec=cur_maskVec, 
                                            grouplasso_module=grouplasso_module, args=args, total_params=total_prunable_params, log_loss=args.log_loss)
        
        # save the training log per epoch -- Optional
        if loss_log != None:
            import json
            with open(f"loss_logs_epoch_{epoch}.json", "w") as log_file:
                json.dump(loss_log, log_file)
        
        # learing rate update
        scheduler_llm.step()
        scheduler_hyper.step()

        if args.tuning_method != 'lora':
            save_fsdp_checkpoint(epoch=epoch, model=llm_ddp, cur_mask_vec=cur_maskVec)
        else:
            save_checkpoint(epoch=epoch, cur_mask_vec=cur_maskVec, model=llm_ddp)

        torch.cuda.empty_cache()
        print(f"cuda cache cleaned for epoch {epoch}")

    print("=====> Training Done. <=====\n")
    training_cleanup()

if __name__ == '__main__':
    main()