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
from hypernet_llm import hard_concrete
import random
import sys
import math
from tqdm import tqdm

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
def match_loss(x, y, epsilon=1e-10):
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

# [ATP_DISP]: user-defined sparsity-level constrain
def caculate_remaining_parmams(pruning_masks, args):
    # version2.0: add [DISP] pruning space parameter counting
    # ** todo: llama3-8b, Qwen2-model param counting
    if args.model == 'llama2-7b':
        assert len(pruning_masks) == 5, 's1-s5 implementation error in [calculate_remaining_params] for [DISP], check the code.'
        m_s1 = pruning_masks[0]
        m_s2 = pruning_masks[1]
        m_s3 = pruning_masks[2]
        m_s4 = pruning_masks[3]
        m_s5 = pruning_masks[4]

        # [ATP_DISP]: 1. calculate q, k, v remaining params based on s1
        dim_after_pruning_qkv_in = torch.sum(m_s1, dim=1)
        remaining_qkv_params     = 3 * torch.sum(4096 * dim_after_pruning_qkv_in)

        # [ATP_DISP]: 2. calculate attn_out remaining params based on s2
        dim_after_pruning_o_out  = torch.sum(m_s2, dim=1)
        remaining_o_params       = torch.sum(4096 * dim_after_pruning_o_out)

        # [ATP_DISP]: 3. calculate mlp_up/gate remaining params based on s3, s4
        dim_after_pruning_mlp_in  = torch.sum(m_s3, dim=1)
        dim_after_pruning_mlp_out = torch.sum(m_s4, dim=1)
        remaining_u_g_params      = 2 * torch.sum(dim_after_pruning_mlp_in * dim_after_pruning_mlp_out)

        # [ATP_DISP]: 4. calculate mlp_down params based on s4, s5
        dim_after_pruning_d_out   = torch.sum(m_s5, dim=1)
        remaining_down_params     = torch.sum(dim_after_pruning_mlp_out * dim_after_pruning_d_out)

        total_remaining_params = remaining_qkv_params + remaining_o_params + remaining_u_g_params + remaining_down_params

        return total_remaining_params
    
    else:

        raise NotImplementedError

#-----------------------------------------------------------------#
# [ATP_DISP]: step_wise target_llm param_tuning based on llm.train_forward()
def target_llm_step(llm_model, input_ids, labels, masks, attn_mask, args):
    # [ATP_DISP]: llm.train_forward() enabled
    llm_model.train()
    
    cur_device = next(llm_model.parameters()).device
    input_ids  = input_ids.to(cur_device)
    attn_mask  = attn_mask.to(cur_device)
    labels     = labels.to(cur_device)

    # [ATP_DISP] 1. llm.train_forward() for NEXT_TOKEN_PREDICTION_LOSS w/ s1-s5 decisions
    if args.tuning_method == "lora":
        output      = llm_model(input_ids=input_ids, 
                        attention_mask=attn_mask,
                        labels=labels, 
                        return_dict=True, 
                        use_cache=False,
                        pruning_mask=masks)
    else:
        output      = llm_model(input_ids=input_ids, 
                                attention_mask=attn_mask,
                                labels=labels, 
                                return_dict=True, 
                                use_cache=False,
                                pruning_mask=None)
    target_loss = output["loss"]
    
    # [ATP_DISP]: the param update would be only constrained by target_loss here :)
    llm_loss = target_loss                         

    # [ATP_DISP]: backward() on the trainable params of the target LLM
    llm_loss.backward()

    return llm_loss
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# [ATP_DISP]: step_wise pruning_decision_generator training based on llm.eval_forward()
def hypernet_step(hypernet, llm_model, val_ids, labels, attn_mask, pruning_ratio_target, total_params, args):
    # [ATP_DISP]: eval_forward() enabled 
    llm_model.eval()
    hypernet.train()

    # [ATP_DISP]: decision generation via hypernet.forward()
    mask_vec = hypernet(dummy=0)                                             # soft_mask fed into llm.eval_forward() for higher stability                       
    binary_mask_vec = hard_concrete(mask_vec)                                # hard_mask for user-defined sparsity level constrain
    assert torch.all(torch.isfinite(mask_vec)), "NaN or Inf in mask_vec"
    mask = hypernet.module.transform_output(mask_vec)            
    binary_mask = hypernet.module.transform_output(binary_mask_vec)          # transform into applicable S1-S5
    assert len(mask) == 5, "the total masking vectors have been wrong in [hypernet_step], please check the implementation"

    # [ATP_DISP]: eval_forward() with current pruning decision 
    
    output      = llm_model(input_ids=val_ids, 
                            labels=labels, 
                            return_dict=True, 
                            use_cache=False,
                            attention_mask=attn_mask,
                            pruning_mask=mask)
    target_loss = output["loss"]
    
    # [ATP_DISP]: constrain current overall sparsity level of the 'pruned' llm
    remaining_params = caculate_remaining_parmams(pruning_masks=binary_mask, args=args)
    mask_ratio       = remaining_params / total_params
    ratio_loss       = match_loss(mask_ratio, pruning_ratio_target)
    
    # [ATP_DISP]: combined loss for PDG update and backward()
    hyper_loss = target_loss + 5 * ratio_loss 
    hyper_loss.backward()

    return hyper_loss, target_loss, ratio_loss
#-----------------------------------------------------------------# 

#-----------------------------------------------------------------#
# [ATP_DISP]: one ATP step based on DISP pruning space 
def llm_sp_train_one_epoch(nlp_dataloader, nlp_hypernet_dataloader, target_llm, hyper_net, optimizer_llm, optimizer_hyper, epoch, total_epochs, cur_mask_vec, grouplasso_module, args, total_params, log_loss):
    print(f"Epoch {epoch} starting.............")
    # [ATP_DISP]: average loss tracker
    llm_loss_ave       = AverageMeter()
    hypernet_loss_ave  = AverageMeter()
    valid_loss_ave     = AverageMeter()
    ratio_loss_ave     = AverageMeter()

    reduced_hyper_loss = None
    reduced_valid_loss = None
    reduced_ratio_loss = None

    # [ATP_DISP]: step-wise loss log -- Optional
    if log_loss:
        loss_logs = {
        "llm_loss": [],
        "group_lasso_loss": [],
        "hypernet_loss": [],
        "valid_loss": [],
        "ratio_loss": [],
        }
    else:
        loss_logs = None
    
    pruning_ratio_target =  1 - args.pruning_ratio_target

    # [ATP_DISP]: deal with 'return_mask' if current PDG is frozen
    if epoch >= (args.start_epoch_control + args.control_epochs):
        print(f"[Pruning_Decision] is pre-fixed, only target LLM weight would be updated in epoch: {epoch}")
        return_mask = copy.deepcopy(cur_mask_vec)
        masks       = hyper_net.module.transform_output(cur_mask_vec)
    else:
        print(f"[Pruning_Decision] is newly-generated, Pruning_Decision_Generator togetherwith target LLM would be updated in epoch: {epoch}")

    # [ATP_DISP]: timer 
    start_time = time.time()

    # [ATP_DISP]: 1. optimize pruning decisions via PDG training
    assert len(nlp_hypernet_dataloader) != 0, "Error: The nlp_hypernet_dataloader is empty."
    nlp_hypernet_iter = itertools.cycle(nlp_hypernet_dataloader)

    ###############################################
    # count total steps 
    total_steps = total_epochs * len(nlp_dataloader)

    for i, text_input in enumerate(tqdm(nlp_dataloader, desc="Processing", unit="batch")):
        if epoch >= args.start_epoch_control and epoch < (args.start_epoch_control + args.control_epochs):
            if (i + 1) % args.control_step == 0:
                # acquire validation mini-batch
                val_inputs = next(nlp_hypernet_iter)
                optimizer_hyper.zero_grad()

                # train hypernet()
                hyper_loss, valid_loss, ratio_loss = hypernet_step(
                    hypernet=hyper_net, 
                    llm_model=target_llm, 
                    val_ids=val_inputs["input_ids"], 
                    attn_mask=val_inputs["attention_mask"], 
                    labels=val_inputs["labels"],
                    pruning_ratio_target=pruning_ratio_target, 
                    total_params=total_params,
                    args=args,
                )

                torch.nn.utils.clip_grad_norm_(hyper_net.parameters(), 1.0)
                optimizer_hyper.step()

                # generate updated s1-s5 for target LLM param update
                # **notice: gumble_sigmoid would bring in slight randomness issues, thus the same s1-s5 would be shared across devices
                # ** w/o broadcast would lead to flat variance of grouplasso loss (tested in version0.1)
                with torch.no_grad():
                    hyper_net.eval()
                    # broadcast the same s1-s5 to all devices
                    if dist.get_rank() == 0:
                        mask_vec = hyper_net(dummy=0)
                    else:
                        mask_vec = torch.empty_like(hyper_net(dummy=0))
                    dist.broadcast(mask_vec, src=0)
                    return_mask = copy.deepcopy(mask_vec)
                    masks = hyper_net.module.transform_output(mask_vec)

                # update average loss tracker
                reduced_hyper_loss = reduce_loss(hyper_loss)
                reduced_valid_loss = reduce_loss(valid_loss)
                reduced_ratio_loss = reduce_loss(ratio_loss)
                
                hypernet_loss_ave.update(reduced_hyper_loss.item(), 1)
                valid_loss_ave.update(reduced_valid_loss.item(), val_inputs["input_ids"].size(0))
                ratio_loss_ave.update(reduced_ratio_loss.item(), 1)

                if log_loss:
                    loss_logs["hypernet_loss"].append(reduced_hyper_loss.item())
                    loss_logs["valid_loss"].append(reduced_valid_loss.item())
                    loss_logs["ratio_loss"].append(reduced_ratio_loss.item())
        
        # after s1-s5 fixed, validation would be periodically performed based on the current llm + fixed pruning decisions
        # ** for tracking training quality purpose only
        if epoch >= (args.start_epoch_control + args.control_epochs):
            if i % args.valid_interval == 0:
                target_llm.eval()
                val_inputs = next(nlp_hypernet_iter)
                with torch.no_grad(): 
                    temp_output = target_llm(
                        input_ids=val_inputs["input_ids"], 
                        labels=val_inputs["labels"], 
                        return_dict=True, 
                        use_cache=False,
                        attention_mask=val_inputs["attention_mask"],
                        pruning_mask=masks
                    )
                    validation_purpose_loss = temp_output["loss"]
                    print(f"The current validation loss with fixed maskSchedule: {validation_purpose_loss}")
                del temp_output
                del validation_purpose_loss

        # [ATP_DISP]: 2. optimize target LLM trainable params
        optimizer_llm.zero_grad()

        current_lr = optimizer_llm.param_groups[0]['lr']
        llm_loss = target_llm_step(
            llm_model=target_llm, 
            input_ids=text_input["input_ids"], 
            labels=text_input["labels"],
            masks=masks, 
            attn_mask=text_input["attention_mask"], 
            args=args, 
        )
        
        torch.nn.utils.clip_grad_norm_(target_llm.parameters(), 1.0)
        optimizer_llm.step()

        # update average loss tracker
        reduced_llm_loss = reduce_loss(llm_loss)
        llm_loss_ave.update(reduced_llm_loss.item(), text_input["input_ids"].size(0))
        
        if log_loss:
            loss_logs["llm_loss"].append(reduced_llm_loss.item())

        # [ATP_DISP]: 3. perform grouplasso approximal projection
        if args.tuning_method == 'lora':
            '''
            if epoch >= (args.start_epoch_control + args.control_epochs):
                grouplasso_module.grad_mul = 1000
            else:
                grouplasso_module.grad_mul = 100000
            '''
            # update: new projection tensity (beta)
            current_step = len(nlp_dataloader) * epoch + i
            current_step_tensor = torch.tensor(current_step, dtype=torch.float32)
            total_steps_tensor = torch.tensor(total_steps, dtype=torch.float32)
            grouplasso_module.grad_mul = (100000 * torch.log(current_step_tensor + 1) / torch.log(total_steps_tensor + 1)).item()
            grouplasso_module.lr = current_lr

            projection_status = grouplasso_module.project_weight_lora_DISP(
                target_llm=target_llm.module, 
                pruning_masks=masks, 
            )
            assert projection_status == True, "weight_projection for [ATP_DISP] failed, check the code."

            if i % args.gltrack_interval == 0:
                gl_loss = grouplasso_module.lora_DISP_forward(target_llm = target_llm.module, pruning_masks = masks)
                print(f"current group lasso loss after projection: {gl_loss}")
        ###############################################

        # [ATP_DISP]: training log
        # **updated: we would log out the current losses, the averaged loss would be only logged when an epoch is done
        if i % args.log_interval == 0:
            if torch.distributed.get_rank() == 0:
                elapsed_time = time.time() - start_time
                print("timer:")
                print(elapsed_time)
                print(
                    f"Time: {elapsed_time:.2f}s | "
                    f"Step: {i} | "
                    f"LLM Loss: {reduced_llm_loss:.3f} | "
                    f"Hypernet Loss: {reduced_hyper_loss:.3f if reduced_hyper_loss is not None else 0.000} | "
                    f"Valid Loss: {reduced_valid_loss:.3f if reduced_valid_loss is not None else 0.000} | "
                    f"Ratio Loss: {reduced_ratio_loss:.3f if reduced_ratio_loss is not None else 0.000} | "
                )

            start_time = time.time()

    # rank0 would log the average summary per epoch
    if torch.distributed.get_rank() == 0:
        print(f"\n===== End of Epoch {epoch} =====")
        print(
            f"Epoch {epoch} Summary:\n"
            f"LLM Loss (Avg): {llm_loss_ave.avg:.4f} | "
            f"Hypernet Loss (Avg): {hypernet_loss_ave.avg:.4f} | "
            f"Ratio Loss (Avg): {ratio_loss_ave.avg:.4f} | "
            f"Valid Loss (Avg): {valid_loss_ave.avg:.4f}\n"
        )

    return return_mask, loss_logs

