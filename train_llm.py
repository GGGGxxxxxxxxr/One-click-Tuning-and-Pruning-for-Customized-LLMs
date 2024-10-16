import torch
import time
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
from alignment_function_llm import Group_Lasso_regularization
import itertools

#-----------------------------------------------------------------#
# average computation for loss
class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# DDP_specific loss reducing
def reduce_loss(loss):
    loss_clone = loss.clone()
    dist.all_reduce(loss_clone, op=dist.ReduceOp.SUM)
    loss_clone /= dist.get_world_size()
    return loss_clone
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# the value matching loss
# **usage:    match_loss(x, y = target_value)
# **function: make x --> closer to target_value
def match_loss(x, y, epsilon=1e-8):
    """
    Computes R(x, y) = log(max(x, y) / (min(x, y) + epsilon)) to avoid division by zero.
    
    Args:
    x (torch.Tensor): Input tensor x.
    y (torch.Tensor): Input tensor y.
    epsilon (float): Small constant for numerical stability to prevent division by zero.
    
    Returns:
    torch.Tensor: The computed loss value.
    """
    # Compute max(x, y) and min(x, y)
    if torch.is_tensor(y):
        y = y
    else:
        y = torch.tensor(y)
    max_val = torch.max(x, y)
    min_val = torch.min(x, y)

    # To avoid division by zero, add a small epsilon to min_val
    ratio = max_val / (min_val + epsilon)

    # Compute log(max / min)
    loss = torch.log(ratio)
    
    return loss
#-----------------------------------------------------------------#

def process_tensor_list(tensor_list):
    concatenated = torch.stack([torch.sum(tensor, dim=1) for tensor in tensor_list], dim=1)
    max_values, max_indices = torch.max(concatenated, dim=1)
    mean_of_max_values = torch.mean(max_values)

    return mean_of_max_values

#-----------------------------------------------------------------#
# loss function for intra-head dimensional alignment 
# ** for maintaining parallel processing ability of GPU, MHA or GA shall be aligned within K, V dimension.
def dim_alignment_loss(mask, num_key_value, match_loss):
    """
    Computes the alignment loss for a given mask.

    Args:
    mask (list): List of masks with length num_key_value.
    num_key_value (int): The number of key-value pairs.
    match_loss (function): A function that calculates the match loss between current mask and average mask.

    Returns:
    alignment_loss (float): The computed alignment loss.
    """
    alignment_loss = 0

    # Concatenate masks along dimension 1
    total_mask = torch.cat(mask, dim=1)
    
    # Compute the average mask
    mask_ave = total_mask.sum(dim=1) / num_key_value

    # Loop through each mask and compute alignment loss
    for head_idx in range(num_key_value):
        cur_masked_dim = mask[head_idx].sum(dim=1)
        alignment_loss += match_loss(cur_masked_dim, mask_ave)
    
    return torch.mean(alignment_loss)
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# step_wise forward() for target_llm param_tuning
def target_llm_step(llm_model, input_ids, masks, attn_mask, epoch, args, gl_module, scaler):
    llm_model.train()
    #uniform device
    cur_device = next(llm_model.parameters()).device
    input_ids = input_ids.to(cur_device)
    attn_mask = attn_mask.to(cur_device)
    seq_len = input_ids.shape[1]

    # a) llm_forward() for NEXT_TOKEN_PREDICTION_LOSS w/o pruning masks
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output      = llm_model(input_ids=input_ids, 
                                attention_mask=attn_mask,
                                labels=input_ids, 
                                return_dict=True, 
                                use_cache=False,
                                num_logits_to_keep=seq_len, 
                                pruning_mask=None)
    target_loss = output["loss"]

    
    # b) if current_epoch >= args.start_epoch_regularization:
    # **Group Lasso Sparsity Regularization is performed on the masked weights.
    # ** we dont use such gl_loss as backward() to update the grouplasso regularization
    # ** we only use it as a value inspector, thus no_grad_fn would be applied here
    # ** GroupLasso is implemented via direct WeightProjection
    if epoch >= args.start_epoch_regularization:
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            gl_loss = gl_module(target_llm = llm_model.module, pruning_masks = masks, epoch=epoch)
    else:
        gl_loss = torch.tensor(0.0).to(target_loss.device)
    
    '''
    ** depreciated for FSDP mode, cuz GroupLassoLoss via backward() would cause severe memory consumption issue for CUDA **
    # c) combined loss for target_llm_param optimization
    # ** adjust tensity for GroupLasso Regularization, when training is close to the end, increase the tensity to make sure that GroupLassoLoss is close to 0.
    if epoch >= (args.epochs - 9):
        gl_tensity = 1000           # force to set expected weights to ZERO
    else: 
        gl_tensity = 1
    '''

    llm_loss = target_loss  #+ gl_tensity * gl_loss

    scaler.scale(llm_loss).backward()

    return llm_loss, target_loss, gl_loss
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# step_wise forward() for hypernet() param_tuning
def hypernet_step(hypernet, llm_model, val_ids, attn_mask, pruning_ratio_target, num_key_value, pruning_contribution):
    # acquire K, V, O, Up mask pruning contributions
    k_ratio = pruning_contribution["k_ratio"]
    v_ratio = pruning_contribution["v_ratio"]
    o_ratio = pruning_contribution["o_ratio"]
    u_ratio = pruning_contribution["u_ratio"]

    # a) freeze llm & unfreeze hypernet()
    llm_model.eval()
    hypernet.train()

    # b) hypernet.forward() (get logits instead of binary mask for hypernet() training)
    # acquire trainable mask for masked_llm inference
    mask_vec = hypernet.module()
    assert torch.all(torch.isfinite(mask_vec)), "NaN or Inf in mask_vec"
    mask = hypernet.module.transform_output(mask_vec)

    # c) masked_llm forward() with 'pruning_mask = mask'
    seq_len = val_ids.shape[1]
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        output      = llm_model(input_ids=val_ids, 
                                labels=val_ids, 
                                return_dict=True, 
                                use_cache=False,
                                num_logits_to_keep=seq_len, 
                                attention_mask=attn_mask,
                                pruning_mask=mask)
    target_loss = output["loss"]

    # d) mask constrain: total pruning ratio + head-wise dimensional alignment
    # i) the total mask ratio is close to 0.5
    # ** modified logic for the real pruning ratio into 0.5, each (0/1) within the maskVec would contribute differently to the total p
    '''
    **deprecated naive mask ratio constrain
    mask_sum    = torch.sum(mask_vec)
    total_count = mask_vec.numel()
    '''
    total_count =   k_ratio   * torch.cat(mask[:num_key_value]).numel() \
                    + v_ratio * torch.cat(mask[num_key_value : 2 * num_key_value]).numel() \
                    + o_ratio * mask[-2].numel() \
                    + u_ratio * mask[-1].numel()
    
    mask_sum    =   k_ratio   * torch.sum(torch.cat(mask[:num_key_value])) \
                    + v_ratio * torch.sum(torch.cat(mask[num_key_value : 2 * num_key_value])) \
                    + o_ratio * torch.sum(mask[-2]) \
                    + u_ratio * torch.sum(mask[-1]) 
    
    mask_ratio  = mask_sum / total_count
    ratio_loss  = match_loss(mask_ratio, pruning_ratio_target)

    # ii) the intra-head dimensional alignment (specifically, for mask_K & mask_V)
    # the first version of implementation is too harsh sometimes, so that the Hypernet() would tend to never prune any of the attention part.
    # we turn to penalize the max() remaining dimension of K, V within the same layer to formulate a softer restriction
    '''
    alignment_loss = 0
    mask_k = mask[:num_key_value]
    alignment_loss += dim_alignment_loss(mask_k, num_key_value, match_loss)
    mask_v = mask[num_key_value: 2*num_key_value]
    alignment_loss += dim_alignment_loss(mask_v, num_key_value, match_loss)
    '''
    alignment_loss = 0
    mask_k = mask[:num_key_value]
    mask_v = mask[num_key_value: 2 * num_key_value]
    alignment_loss += process_tensor_list(mask_k)
    alignment_loss += process_tensor_list(mask_v)

    # e) sum the loss
    hyper_loss = target_loss + 10 * ratio_loss + 0.002 * alignment_loss

    hyper_loss.backward()

    with torch.no_grad():
        hypernet.eval()
        mask_vec    = hypernet.module()  
        return_mask = copy.deepcopy(mask_vec)
        masks       = hypernet.module.transform_output(mask_vec)

    return hyper_loss, target_loss, ratio_loss, alignment_loss, masks
#-----------------------------------------------------------------# 

#-----------------------------------------------------------------#
# llm_structured_pruning training for one_epoch
# step1: freeze hypernet(), generate pruning [pruning_MASK]; 
# (if epoch > hyper_end, then use the pre-fixed [pruning_MASK], the following epochs would only do further param tuning based on finalized [pruning_MASK])
# step2: unfreeze target llm, param tuning (NO MASK) based on specific dataset via CrossEntropyLoss() for TARGET_LOSS;
# step3: if epoch in [gl_start, gl_end], group_lasso in target_llm_weights based on [pruning_MASK];
# step4: if epoch in [hyper_start, hyper_end], freeze target llm, unfreeze hypernet(), perform weight update on hypernet() via MASKED_LLM_TARGET_LOSS + LOSS(model_size)
# return [pruning_MASK] 

# ** logging info is printed out per 20 steps.
# ** GroupLasso is applied for WeightProjection (directly imported from ATO's raw implementation)
# ** [Calibration_Dataset] for hypernet() training is a small portion of data from the NLP dataset['train']
# ** Casual LLM takes shifted [input_ids] as the self-supervised training labels for NEXT_TOKEN_PREDICTION
def llm_sp_train_one_epoch(nlp_dataloader, nlp_hypernet_dataloader, target_llm, hyper_net, optimizer_llm, optimizer_hyper, epoch, cur_mask_vec, grouplasso_module, args, scaler, pruning_contribution):
    print(f"Epoch {epoch} starting.............")
    # initialize training loss holder
    llm_loss_ave       = AverageMeter()
    target_loss_ave    = AverageMeter()
    gl_loss_ave        = AverageMeter()
    hypernet_loss_ave  = AverageMeter()
    valid_loss_ave     = AverageMeter()
    ratio_loss_ave     = AverageMeter()
    alignment_loss_ave = AverageMeter()

    pruning_ratio_target = args.pruning_ratio_target
    num_key_value        = args.num_key_values

    # step1: [pruning_MASK] selection (pre-fixed or newly-generated)
    if epoch >= (args.start_epoch_control + args.control_epochs):
        print(f"[Pruning_MASK] is pre-fixed, only target LLM weight would be updated in epoch: {epoch}")
        return_mask = copy.deepcopy(cur_mask_vec)
        masks       = hyper_net.module.transform_output(cur_mask_vec)
    else:
        print(f"[Pruning_MASK] is newly-generated, Hypernet() togetherwith target LLM weight would be updated in epoch: {epoch}")
        with torch.no_grad():
            hyper_net.eval()
            mask_vec    = hyper_net.module()  
            return_mask = copy.deepcopy(mask_vec)
            masks       = hyper_net.module.transform_output(mask_vec)

    # Timer to measure elapsed time
    start_time = time.time()

    assert len(nlp_hypernet_dataloader) != 0, "Error: The nlp_hypernet_dataloader is empty."
    nlp_hypernet_iter = itertools.cycle(nlp_hypernet_dataloader)
    for i, text_input in enumerate(nlp_dataloader):
        # step2： llm param domain-specific tuning with [pruning_MASK] (temporary static)
        optimizer_llm.zero_grad()
        current_lr = optimizer_llm.param_groups[0]['lr']
        llm_loss, target_loss, gl_loss = target_llm_step(llm_model=target_llm, input_ids=text_input["input_ids"], masks=masks, attn_mask=text_input["attention_mask"], epoch=epoch, args=args, gl_module=grouplasso_module, scaler=scaler)
        scaler.unscale_(optimizer_llm) 
        torch.nn.utils.clip_grad_norm_(target_llm.parameters(), 1.0)
        scaler.step(optimizer_llm)
        scaler.update()
        #optimizer_llm.step()

        # update loss into training progress holder
        reduced_llm_loss    = reduce_loss(llm_loss)
        reduced_target_loss = reduce_loss(target_loss)
        reduced_gl_loss     = reduce_loss(gl_loss)
        llm_loss_ave.update(reduced_llm_loss.item(), text_input["input_ids"].size(0))
        target_loss_ave.update(reduced_target_loss.item(), text_input["input_ids"].size(0))
        gl_loss_ave.update(reduced_gl_loss.item(), 1)

        ###############################################
        ### weight_project for grouplasso starting here
        projection_status = grouplasso_module.project_weight(target_llm = target_llm.module, pruning_masks = masks, epoch=epoch, lr=current_lr)
        if projection_status != True:
            print("weight_projection failed, check the code.")
        ###############################################

        # step3: mask_generator(hypernet()) param tuning if epoch >= args.start_epoch_hyper
        if epoch >= args.start_epoch_control and epoch < (args.start_epoch_control + args.control_epochs):
            if (i + 1) % args.control_step == 0:
                # acquire datapiece from 'validation' set for hypernet()
                val_inputs = next(nlp_hypernet_iter)
                # hypernet() param tuning
                optimizer_hyper.zero_grad()
                hyper_loss, valid_loss, ratio_loss, alignment_loss, masks = hypernet_step(hypernet=hyper_net, llm_model=target_llm, val_ids=val_inputs["input_ids"], attn_mask=val_inputs["attention_mask"],pruning_ratio_target=pruning_ratio_target, num_key_value=num_key_value, pruning_contribution=pruning_contribution)
                torch.nn.utils.clip_grad_norm_(hyper_net.parameters(), 1.0)
                optimizer_hyper.step()

                # update loss into training progress holder
                reduced_hyper_loss = reduce_loss(hyper_loss)
                reduced_valid_loss = reduce_loss(valid_loss)
                reduced_ratio_loss = reduce_loss(ratio_loss)
                reduced_align_loss = reduce_loss(alignment_loss)
                hypernet_loss_ave.update(reduced_hyper_loss.item(), 1)
                valid_loss_ave.update(reduced_valid_loss.item(), val_inputs["input_ids"].size(0))
                ratio_loss_ave.update(reduced_ratio_loss.item(), 1)
                alignment_loss_ave.update(reduced_align_loss.item(), 1)

        # periodical training log
        # only print in the master process
        if i % args.log_interval == 0:
            if torch.distributed.get_rank() == 0:
                elapsed_time = time.time() - start_time
                print(f"Time: {elapsed_time:.2f}s | "
                    f"Step: {i} | "
                    f"LLM Loss: {llm_loss_ave.avg:.4f} | "
                    f"Target Loss: {target_loss_ave.avg:.4f} | "
                    f"Group Lasso Loss: {gl_loss_ave.avg:.4f} | "
                    f"Hypernet Loss: {hypernet_loss_ave.avg:.4f} | "
                    f"Valid Loss: {valid_loss_ave.avg:.4f} | "
                    f"Ratio Loss: {ratio_loss_ave.avg:.4f} | "
                    f"Alignment Loss: {alignment_loss_ave.avg:.4f}")
                start_time = time.time()

    # Print final summary for the epoch
    # only print in the master process
    if torch.distributed.get_rank() == 0:
        print(f"\n===== End of Epoch {epoch} =====")
        print(f"Epoch {epoch} Summary:")
        print(f"LLM Loss (Avg): {llm_loss_ave.avg:.4f} | "
            f"Target Loss (Avg): {target_loss_ave.avg:.4f} | "
            f"Group Lasso Loss (Avg): {gl_loss_ave.avg:.4f} | "
            f"Hypernet Loss (Avg): {hypernet_loss_ave.avg:.4f} | "
            f"Ratio Loss (Avg): {ratio_loss_ave.avg:.4f} | "
            f"Alignment Loss (Avg): {alignment_loss_ave.avg:.4f}\n")
    
    return return_mask