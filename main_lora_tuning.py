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

mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    _module_wrap_policy,
    enable_wrap,
    wrap,
)

# 8bit optimizer for memory efficent training
import bitsandbytes as bnb

# llm-related library import
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoftQConfig, LoraConfig, get_peft_model
from util_llm import count_llm_p_structures
from util_llm import pruning_ratio_contribution
from hypernet_llm import LLM_HyperStructure
from train_llm import llm_sp_train_one_epoch, llm_tuning_train_one_epoch
from build_dataset import formatted_MedNLI_dataset, formatted_wikitext_dataset, formatted_AGNews_dataset, create_medical_dataset
# mask_infused_custom_llm
from custom_llms.qwen2 import Qwen2ForCausalLM
from custom_llms.llama import LlamaForCausalLM
from alignment_function_llm import Group_Lasso_regularization
from custom_llms.llama import LlamaDecoderLayer

''''
llama_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )
'''

llama_auto_wrap_policy = partial(
            _module_wrap_policy,
            module_classes ={
                LlamaDecoderLayer,
                nn.Embedding,
                nn.Linear
            }
        )

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
parser.add_argument('--bsz', default=2, type=int,metavar='N',
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
parser.add_argument('--log-interval', default=20, type=int,
                    help='training log intervals of STEPs')
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# methdology-related args
parser.add_argument('--dataset', default='MedNLI', type=str,
                    help='specify the domain-specifc dataset for LLM pararm tuning, **build-in ready2use selections: wikitext, MedNLI, AGNews')
parser.add_argument('--pruning-method', default='inner', type=str,
                    help='head-wise MHA pruning **[head_wise], layer-aware-uniform head pruning **[layer_uniform_attn] or more fine-grained within attention-head pruning **[inner]')
parser.add_argument('--tuning-method',  default='full', type=str,
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
    model=None, 
    filename="/orange/yonghui.wu/sgao1/llm_base_tuning_lora.pth.tar"
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
    state = {'epoch': epoch}

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

def save_fsdp_checkpoint(epoch, model, filename="/orange/yonghui.wu/sgao1/llm_base_tuning_trail.pth.tar"):
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
            state = {
                'model_state_dict': cpu_state,
            }

            # Save only on the main process to avoid multiple processes writing the file
            if torch.distributed.get_rank() == 0:
                torch.save(state, filename)
                print(f"Checkpoint saved at epoch {epoch} to {filename}\n")
    
    dist.barrier()
#-----------------------------------------------------------------#
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def load_checkpoint(model, hyper_net, optimizer_llm, optimizer_hyper, filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))  # adjust map_location as needed
    model.load_state_dict(checkpoint['model_state_dict'])
    hyper_net.load_state_dict(checkpoint['hyper_net_state_dict'])
    optimizer_llm.load_state_dict(checkpoint['optimizer_llm_state_dict'])
    optimizer_hyper.load_state_dict(checkpoint['optimizer_hyper_state_dict'])
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
    
    print("Welcome to the ATO's one-step tuning & structural pruning for LLMs!\n")
    print("=====> DDP Training ENV Initialization Done. <=====\n")
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
    # ** current support: {qwen2-0.5b, llama2-7b}
    print("=====> Intialization pre-trained: '{}' from Huggingface <=====".format(args.model))
    if args.model == 'qwen-0.5b':
        model_cfg = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        model     = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to(device)
        args.num_key_values = model_cfg.num_key_value_heads

        print("=====> Model structure: <=====")
        print(model)
        print("=====> Structure End. <=====\n")

    # llama2-7b initialization from Huggingface
    # ** llama tokenizer does not have a specific PAD token, so a special pad token is appended here for string length alignment
    elif args.model == 'llama2-7b':
        api_token = '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'
        model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf",  token = api_token)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token = api_token)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model     = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="sdpa", token = api_token).to(device)
        model.resize_token_embeddings(len(tokenizer))
        args.num_key_values = model_cfg.num_key_value_heads
    else:
        print("=====> Model not implemented yet! System Exit. <=====\n")
        sys.exit()
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # LoRA integration or full-param tuning
    # currently we use full-param, LoRA feature would be developed later
    if args.tuning_method == 'lora':
        print("=====> LoRA Tuning Initialization, pre-trained weights would be frozen during the following stages. <=====\n")
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
        print("=====> LoRA infusion done. <=====\n")
        model.print_trainable_parameters()
    else:
        print("=====> Full-param tuning Initialization Done. Pre-trained weights would be tuned during the following stages. <=====\n")
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # dataset initialization
    if args.dataset == "wikitext":
        nlp_dataset, val_dataset = formatted_wikitext_dataset()
    elif args.dataset == 'MedNLI':
        nlp_dataset, val_dataset = formatted_MedNLI_dataset()
    elif args.dataset == 'AGNews':
        nlp_dataset, val_dataset = formatted_AGNews_dataset()
    elif args.dataset == 'medical':
        nlp_dataset, val_dataset = create_medical_dataset()

    print("=====> Dataset Config & Sample Check: <=====\n")
    print(nlp_dataset)
    print(val_dataset)

    # Print the 85th samples
    print("=====> The 85th Sequence Sample: <=====")
    print(nlp_dataset[85])  
    print(val_dataset[85])
    print("=====> Tokenized 85th Sequence Sample: <=====")
    # tokenize the NLP dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='longest')
    
    tokenized_datasets = nlp_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
    tokenized_valsets  = val_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
    print(tokenized_datasets[85])
    print(tokenized_valsets[85])
    print("=====> NLP Dataset Initialization Done. <=====")
    # config training dataloader
    data_collator  = DataCollatorWithPadding(tokenizer=tokenizer)
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
    print("=====> NLP DDP_DataLoader Initialization Done. <=====\n")
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # Training Optimizier (llm target loss is automatically calculated within the CasualLM.forward() with CrossEntropyLoss(), we only need to take care of the GroupLasso and HyperNet() update)
    # 1. For both target LLM and HyperNet(), AdamW() is applied as the param optimizer
    # 2. For HyperNet() and target LLM, common CosLR is applied for learningrate adjusting for smooth LR dropping
    # Pre-training from scratch probably needs CosWarmRestart() but as we have a inital point from pre-trained, we could go for CosLR directly.

    #llm_ddp        = DDP(model, device_ids=[device])
    if args.tuning_method == 'lora':
        print("DDP has been launched for LoRA tuning.")
        llm_ddp        = DDP(model, device_ids=[device])
    else:
        print("FSDP has been launched for Full-param tuning in case of CUDA OOM issue.")
        llm_ddp        = FSDP(model, device_id=device, auto_wrap_policy=llama_auto_wrap_policy, use_orig_params=False, mixed_precision=mixed_precision_policy)

    llm_params      = llm_ddp.parameters()

    if args.use_8bit_training == True:
        optimizer_llm   = bnb.optim.AdamW8bit(llm_params,lr = args.lr)
    else:
        optimizer_llm   = torch.optim.AdamW(llm_params,lr = args.lr)

    scheduler_llm   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_llm, T_max=args.epochs, eta_min=1e-6)

    print("=====> Trainable parameters for target_LLM: <=====")
    for name, param in llm_ddp.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")

    print("=====> Training Optimizers and Schedulers Initialization Done. <=====\n")
    
    #-----------------------------------------------------------------#
    if args.tuning_method != "lora":
        print("schardedGradScaler has been intialized for FSDP.")
        scaler = ShardedGradScaler()
    else:
        print("AMP is initialized for LoRA Finetuning.")
        scaler = torch.amp.GradScaler()
        
    #-----------------------------------------------------------------#
    # Training Process
    print("=====> Begin Training: <=====\n")

    print(f"=====> New Training Progress Launched. <=====\n")
    start_epoch = args.start_epoch

    for epoch in range(start_epoch, args.epochs):

        # data shuffle epoch-wisely
        ddp_sampler.set_epoch(epoch)
        ddp_sampler1.set_epoch(epoch)

        # train for one epoch
        training_status = llm_tuning_train_one_epoch(nlp_dataloader=nlp_dataloader, target_llm=llm_ddp, optimizer_llm=optimizer_llm, epoch=epoch, args=args, scaler=scaler)
        
        # learing rate update
        scheduler_llm.step()

        if args.tuning_method == "lora":
            save_checkpoint(epoch=epoch, model=llm_ddp)
        else:
            save_fsdp_checkpoint(epoch=epoch, model=llm_ddp)
    
        torch.cuda.empty_cache()
        print(f"cuda cache cleaned for epoch {epoch}")

    print("=====> Training Done. <=====\n")
    training_cleanup()

if __name__ == '__main__':
    main()