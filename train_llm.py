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

def caculate_remaining_parmams(pruning_masks, args):
    if args.model == 'llama2-7b':
        assert len(pruning_masks) == 3, 'pruning masks implementation error in [calculate_remaining_params], check the code.'
        m_K = pruning_masks[0]
        m_V = pruning_masks[1]
        m_out = pruning_masks[2]

        # calculate q_proj, k_proj remaining params
        # input dim is 4096 because the hidden_states of each layer is unpruned, as [down_proj] has no output pruning masks
        # m_K [32, 4096] (repeated head-wise pruning mask [32, 128] for 32 times to expand into the linear mask)
        dim_after_pruning_K_out = torch.sum(m_K, dim=1)
        remaining_K_params  = 4096 * dim_after_pruning_K_out
        remaining_QK_params = torch.sum(remaining_K_params * 2)

        # calculate v_proj remaining params
        dim_after_pruning_V_out = torch.sum(m_V, dim=1)
        remaining_V_params = torch.sum(4096 * dim_after_pruning_V_out)

        # calculate out_proj remaining params
        remaining_out_params = torch.sum(4096 * dim_after_pruning_V_out)

        # calculate mlp_up / gate remaining params
        dim_after_pruning_up_out = torch.sum(m_out, dim=1)
        remaining_up_gate_params = 2 * torch.sum(4096 * dim_after_pruning_up_out)

        # calculate mlp_down remaining params
        remaining_down_params = torch.sum(4096 * dim_after_pruning_up_out)

        total_remaining_params = remaining_QK_params + remaining_V_params + remaining_out_params + remaining_up_gate_params + remaining_down_params

        return total_remaining_params
    
    # UPDATED IN VERSION 0.2.1
    # we further added the unpruned parameters (embeddings, lm_head, ...) to achieve more accurate pruning control
    elif args.model == "llama3-8b":
        assert len(pruning_masks) == 3, 'pruning masks implementation error in [calculate_remaining_params], check the code.'
        m_K = pruning_masks[0] 
        m_V = pruning_masks[1]
        m_out = pruning_masks[2]

        m_Q = m_K.repeat(1,4)
        m_attn_out = m_V.repeat(1,4)
        assert m_K.shape == (32, 1024), f"Expected shape of m_K (32, 1024), but got {m_K.shape}"
        assert m_Q.shape == (32, 4096), f"Expected shape of m_Q (32, 4096), but got {m_Q.shape}"
        assert m_attn_out.shape == (32, 4096), f"Expected shape of m_Out (32, 14336), but got {m_attn_out.shape}"
        assert m_out.shape == (32, 14336), f"Expected shape of m_Out (32, 14336), but got {m_out.shape}"

        # calculate q_proj, k_proj remaining params
        # input dim is 4096 because the hidden_states of each layer is unpruned, as [down_proj] has no output pruning masks
        # m_K [32, 4096] (repeated head-wise pruning mask [32, 128] for 32 times to expand into the linear mask)
        dim_after_pruning_K_out = torch.sum(m_K, dim=1)
        remaining_K_params  = 4096 * dim_after_pruning_K_out
        remaining_K_params = torch.sum(remaining_K_params)
        dim_after_pruning_Q_out = torch.sum(m_Q, dim=1)
        remaining_Q_params = 4096 * dim_after_pruning_Q_out
        remaining_Q_params = torch.sum(remaining_Q_params)

        # calculate v_proj remaining params
        dim_after_pruning_V_out = torch.sum(m_V, dim=1)
        remaining_V_params = torch.sum(4096 * dim_after_pruning_V_out)

        # calculate out_proj remaining params
        dim_after_pruning_out_in = torch.sum(m_attn_out, dim=1)
        remaining_out_params     = torch.sum(4096 * dim_after_pruning_out_in)

        # calculate mlp_up / gate remaining params
        dim_after_pruning_up_out = torch.sum(m_out, dim=1)
        remaining_up_gate_params = 2 * torch.sum(4096 * dim_after_pruning_up_out)

        # calculate mlp_down remaining params
        remaining_down_params  = torch.sum(4096 * dim_after_pruning_up_out)

        total_remaining_params = remaining_K_params + remaining_Q_params + remaining_V_params + remaining_out_params + remaining_up_gate_params + remaining_down_params

        return total_remaining_params
    
    else:
        raise NotImplementedError

        

#-----------------------------------------------------------------#
# step_wise forward() for target_llm param_tuning
def target_llm_step(llm_model, input_ids, labels, masks, attn_mask, epoch, args, gl_module, scaler):
    llm_model.train()
    #uniform device
    cur_device = next(llm_model.parameters()).device
    input_ids = input_ids.to(cur_device)
    attn_mask = attn_mask.to(cur_device)
    labels    = labels.to(cur_device)
    seq_len = input_ids.shape[1]

    # a) llm_forward() for NEXT_TOKEN_PREDICTION_LOSS w/o pruning masks
    #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    if args.tuning_method == "lora":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output      = llm_model(input_ids=input_ids, 
                            attention_mask=attn_mask,
                            labels=labels, 
                            return_dict=True, 
                            use_cache=False,
                            #num_logits_to_keep=seq_len, 
                            pruning_mask=masks)
    else:
        output      = llm_model(input_ids=input_ids, 
                                attention_mask=attn_mask,
                                labels=labels, 
                                return_dict=True, 
                                use_cache=False,
                                #num_logits_to_keep=seq_len, 
                                pruning_mask=None)
    target_loss = output["loss"]
    
    
    # b) if current_epoch >= args.start_epoch_regularization:
    # **Group Lasso Sparsity Regularization is performed on the masked weights.
    # ** we dont use such gl_loss as backward() to update the grouplasso regularization
    # ** we only use it as a value inspector, thus no_grad_fn would be applied here
    # ** GroupLasso is implemented via direct WeightProjection

    if epoch >= args.start_epoch_regularization:
        if epoch == (args.epochs - 1):
            gl_tensity = 100000                              # force to set expected weights to ZERO
            gl_module.grad_mul = gl_tensity
        else: 
            gl_tensity = 0.3
            gl_module.grad_mul = gl_tensity

    if args.tuning_method != 'lora':
        if epoch >= args.start_epoch_regularization:
            #with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            gl_loss = gl_module(target_llm = llm_model.module, pruning_masks = masks, epoch=epoch)
        else:
            gl_loss = torch.tensor(0.0).to(target_loss.device)
    else:
        if epoch >= args.start_epoch_regularization:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gl_loss = gl_module.lora_forward(target_llm = llm_model.module, pruning_masks = masks)
        else:
            gl_loss = torch.tensor(0.0).to(target_loss.device)

    
    #** depreciated for FSDP mode, cuz GroupLassoLoss via backward() would cause severe memory consumption issue for CUDA **
    # c) combined loss for target_llm_param optimization
    # ** adjust tensity for GroupLasso Regularization, when training is close to the end, increase the tensity to make sure that GroupLassoLoss is close to 0.
    '''
    if epoch == (args.epochs - 1):
        gl_tensity = 1000                              # force to set expected weights to ZERO
    else: 
        gl_tensity = 1
    '''

    if args.tuning_method != 'lora':
        llm_loss = target_loss                         # in FSDP mode, we are forced to use GroupLasso DirectProjection to simulate such GL_loss backward effects
    else:
        llm_loss = target_loss + gl_loss

    scaler.scale(llm_loss).backward()

    '''
    test & track purpose
    '''
    if args.tuning_method != 'lora':
        print(f"group lasso loss before projection: {gl_loss}")

    return llm_loss, target_loss, gl_loss
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# step_wise forward() for hypernet() param_tuning
def hypernet_step(hypernet, llm_model, val_ids, labels, attn_mask, pruning_ratio_target, total_params, args):
    '''
    ** depreciated version.0.1, the pruning ratio is now calculated via more accurate [remaining_parameters] / [total_params]
    ** previous implementation considers the PruningContribution of each [0] within different masking locations
    ** We have encountered the issue that it is not that accurate and cannot satisfy accurate control.
    # acquire K, V, O, Up mask pruning contributions
    k_ratio = pruning_contribution["k_ratio"]
    v_ratio = pruning_contribution["v_ratio"]
    o_ratio = pruning_contribution["o_ratio"]
    u_ratio = pruning_contribution["u_ratio"]  
    '''

    # a) freeze llm & unfreeze hypernet()
    llm_model.eval()
    hypernet.train()

    # b) hypernet.forward() (get logits instead of binary mask for hypernet() training)
    # acquire trainable mask for masked_llm inference
    # *** use soft mask here for llm_structural_detection (no {hardconcrete} to binary)
    #with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
    mask_vec = hypernet(dummy=0)                                                             #.module()
    binary_mask_vec = hard_concrete(mask_vec)
    assert torch.all(torch.isfinite(mask_vec)), "NaN or Inf in mask_vec"
    mask = hypernet.module.transform_output(mask_vec)
    binary_mask = hypernet.module.transform_output(binary_mask_vec)
    assert len(mask) == 3, "the total masking vectors have been wrong in [hypernet_step], please check the implementation"

    # c) masked_llm forward() with 'pruning_mask = mask'
    seq_len = val_ids.shape[1]

    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        output      = llm_model(input_ids=val_ids, 
                                labels=labels, 
                                return_dict=True, 
                                use_cache=False,
                                #num_logits_to_keep=seq_len, 
                                attention_mask=attn_mask,
                                pruning_mask=mask)
    target_loss = output["loss"]
    
    # ** constrain the pruning target
    # d) mask constrain: total pruning ratio + head-wise dimensional alignment
    # i) the total mask ratio is close to 0.5
    # ** modified logic for the real pruning ratio into 0.5
    # ** [remaining_params] / [total_params]
    # ** we use hard_concrete here to acquire more accurate way of real_pruning simulation
    '''
    # DEPRECIATED in VERSION.0.2 UPDATE
    # NO LONGER USE contribution to constrain pruning sparsity
    total_count =   k_ratio   * torch.cat(mask[:num_key_value]).numel() \
                    + v_ratio * torch.cat(mask[num_key_value : 2 * num_key_value]).numel() \
                    + o_ratio * mask[-2].numel() \
                    + u_ratio * mask[-1].numel()
    
    mask_sum    =   k_ratio   * torch.sum(torch.cat(mask[:num_key_value])) \
                    + v_ratio * torch.sum(torch.cat(mask[num_key_value : 2 * num_key_value])) \
                    + o_ratio * torch.sum(mask[-2]) \
                    + u_ratio * torch.sum(mask[-1]) 
    
    mask_ratio  = mask_sum / total_count
    '''
    remaining_params = caculate_remaining_parmams(pruning_masks=binary_mask, args=args)
    mask_ratio       = remaining_params / total_params
    ratio_loss       = match_loss(mask_ratio, pruning_ratio_target)


    # ii) the intra-head dimensional alignment (specifically, for mask_K & mask_V)
    # the first version of implementation is too harsh sometimes, so that the Hypernet() would tend to never prune any of the attention part.
    # we turn to penalize the max() remaining dimension of K, V within the same layer to formulate a softer restriction
    '''
    # DEPRECIATED AFTER VERISON.0.2 UPDATE
    ** The K(Q) mask and V mask are formulated into [layer_count, 4096 (num_head * head_dim)] instead of num_head * [layer_count, head_dim]
    alignment_loss = 0
    mask_k = mask[:num_key_value]
    alignment_loss += dim_alignment_loss(mask_k, num_key_value, match_loss)
    mask_v = mask[num_key_value: 2*num_key_value]
    alignment_loss += dim_alignment_loss(mask_v, num_key_value, match_loss)
    
    alignment_loss = 0
    mask_k = mask[:num_key_value]
    mask_v = mask[num_key_value: 2 * num_key_value]
    assert len(mask_v) + len(mask_k) + 1 == len(mask), "error for extracting binary mask, please check."
    
    alignment_loss += process_tensor_list(mask_k)
    alignment_loss += process_tensor_list(mask_v)
    '''
    # [32, 4096] for llama2-7b and [32, 1024] for llama3-8b (to be more accurate, it is [32, 128] repeated num_kv_head times)
    mask_k = binary_mask[0] 
    mask_v = binary_mask[1]
    remaining_K_out_dim     = torch.sum(mask_k, dim=1)  #[32,]
    remaining_V_out_dim     = torch.sum(mask_v, dim=1)  #[32,]
    max_remaining_K_out_dim = torch.max(remaining_K_out_dim)
    max_remaining_V_out_dim = torch.max(remaining_V_out_dim)
    alignment_loss          = (max_remaining_K_out_dim + max_remaining_V_out_dim) / args.num_key_values
    
    # e) sum the loss
    hyper_loss = target_loss + 5 * ratio_loss #+ 0.00005 * alignment_loss

    hyper_loss.backward()

    #scaler.scale(hyper_loss).backward()
    '''
    with torch.no_grad():
        hypernet.eval()
        mask_vec    = hypernet.module()  
        return_mask = copy.deepcopy(mask_vec)
        masks       = hypernet.module.transform_output(mask_vec)
    '''

    return hyper_loss, target_loss, ratio_loss, alignment_loss #, mask
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
def llm_sp_train_one_epoch(nlp_dataloader, nlp_hypernet_dataloader, target_llm, hyper_net, optimizer_llm, optimizer_hyper, epoch, cur_mask_vec, grouplasso_module, args, scaler, scaler_hyper, total_params, skip_hyper_training):
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

    # 添加计数器和标志
    ratio_loss_counter = 0  # 用于计数 ratio_loss 连续小于阈值的次数
    ratio_loss_threshold = 0.005                 # 0.005 equals to a gap range of 0.5%
    ratio_loss_consecutive_steps = 4000
    skip_hypernet_training = skip_hyper_training  # 标志：是否跳过 hypernet 的训练

    gl_loss_counter = 0
    gl_loss_threshold = 1.57
    gl_loss_consecutive_steps = 1000000   # if you are using lora, we recommend using epoch instead of specific step control, as overfitting is not that much heavy an issue
    terminate_training = False
    
    print(f"skip_hypernet_training_status: {skip_hypernet_training}")

    hyper_lr_reduced = False

    # step1: [pruning_MASK] selection (pre-fixed or newly-generated)
    if epoch >= (args.start_epoch_control + args.control_epochs) or skip_hypernet_training:
        print(f"[Pruning_MASK] is pre-fixed, only target LLM weight would be updated in epoch: {epoch}")
        return_mask = copy.deepcopy(cur_mask_vec)
        masks       = hyper_net.module.transform_output(cur_mask_vec)
    else:
        print(f"[Pruning_MASK] is newly-generated, Hypernet() togetherwith target LLM weight would be updated in epoch: {epoch}")
        '''
        with torch.no_grad():
            hyper_net.eval()
            mask_vec    = hyper_net(dummy=0)                                               #.module()  
            return_mask = copy.deepcopy(mask_vec)
            masks       = hyper_net.module.transform_output(mask_vec)
        '''

    # Timer to measure elapsed time
    start_time = time.time()

    assert len(nlp_hypernet_dataloader) != 0, "Error: The nlp_hypernet_dataloader is empty."
    nlp_hypernet_iter = itertools.cycle(nlp_hypernet_dataloader)
    for i, text_input in enumerate(nlp_dataloader):
        # Step 1: Hypernet 训练及生成新 Mask
        if epoch >= args.start_epoch_control and epoch < (args.start_epoch_control + args.control_epochs) and not skip_hypernet_training:
            if (i + 1) % args.control_step == 0:
                # 从验证集获取数据用于 Hypernet 的训练
                val_inputs = next(nlp_hypernet_iter)
                optimizer_hyper.zero_grad()

                # 调用 hypernet_step 进行超网训练
                hyper_loss, valid_loss, ratio_loss, alignment_loss = hypernet_step(
                    hypernet=hyper_net, 
                    llm_model=target_llm, 
                    val_ids=val_inputs["input_ids"], 
                    attn_mask=val_inputs["attention_mask"], 
                    labels=val_inputs["labels"],
                    pruning_ratio_target=pruning_ratio_target, 
                    #num_key_value=num_key_value, 
                    total_params=total_params,
                    args=args,
                    #scaler=scaler_hyper
                )

                torch.nn.utils.clip_grad_norm_(hyper_net.parameters(), 3.0)
                optimizer_hyper.step()
                
                ## ** UPDATED in Verison 0.2, precise control of pruning mask
                ## ** avoid the high variance of the generated mask due to different batches
                if epoch ==  (args.start_epoch_control + args.control_epochs - 1) and not hyper_lr_reduced:
                    total_iters = len(nlp_dataloader)
                    threshold = int((2/3) * total_iters)
                    if (i + 1) >= threshold:
                        current_lrs = [param_group['lr'] for param_group in optimizer_hyper.param_groups]
                        print(f"Epoch {epoch}, Iteration {i + 1}: current lr of hypernet: {current_lrs}. Reduce optimizer_hyper's learning to 1/5.")
                        
                        # 更新学习率为之前的 1/4
                        for param_group in optimizer_hyper.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.2
                        
                        # 设置标志，确保只执行一次
                        hyper_lr_reduced = True

                # 生成新掩码供 LLM 训练使用
                with torch.no_grad():
                    hyper_net.eval()
                    #mask_vec = hyper_net(dummy=0)   #.module()  
                    # 只在 rank 0 生成 mask_vec
                    if dist.get_rank() == 0:
                        mask_vec = hyper_net(dummy=0)
                    else:
                        # 在其他 rank 处创建一个相同形状的 tensor 占位
                        mask_vec = torch.empty_like(hyper_net(dummy=0))
                    #广播 mask_vec 从 rank 0 到所有其他 GPU
                    dist.broadcast(mask_vec, src=0)
                    return_mask = copy.deepcopy(mask_vec)
                    masks = hyper_net.module.transform_output(mask_vec)

                # 更新超网损失到训练记录
                reduced_hyper_loss = reduce_loss(hyper_loss)
                reduced_valid_loss = reduce_loss(valid_loss)
                reduced_ratio_loss = reduce_loss(ratio_loss)
                reduced_align_loss = reduce_loss(alignment_loss)
                hypernet_loss_ave.update(reduced_hyper_loss.item(), 1)
                valid_loss_ave.update(reduced_valid_loss.item(), val_inputs["input_ids"].size(0))
                ratio_loss_ave.update(reduced_ratio_loss.item(), 1)
                alignment_loss_ave.update(reduced_align_loss.item(), 1)

                # ** automatic determination of stop hypernet() training
                if reduced_ratio_loss.item() <= ratio_loss_threshold:
                    ratio_loss_counter +=1
                else:
                    ratio_loss_counter = 0      #reset
                
                if ratio_loss_counter >= ratio_loss_consecutive_steps:
                    skip_hypernet_training = True
                    print(f"ratio_loss has been below {ratio_loss_threshold} for {ratio_loss_consecutive_steps} steps.")
                    print("Skipping hypernet training, setting grouplasso.lam to 2000 if executing GroupLassoProjection, and fixing mask_vec for future training.")
        
        # if hypernet() is frozen prior to the configured stop epoch, validation would be performed on the current model + fixed masks
        # ** for tracking training purpose only
        if skip_hypernet_training:
            target_llm.eval()
            val_inputs = next(nlp_hypernet_iter)
            with torch.no_grad():  # 添加 no_grad 以减少内存占用
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    temp_output = target_llm(
                        input_ids=val_inputs["input_ids"], 
                        labels=val_inputs["input_ids"], 
                        return_dict=True, 
                        use_cache=False,
                        num_logits_to_keep=val_inputs["input_ids"].shape[1], 
                        attention_mask=val_inputs["attention_mask"],
                        pruning_mask=masks
                    )
                validation_purpose_loss = temp_output["loss"]
                print(f"The current validation loss with fixed maskSchedule: {validation_purpose_loss}")
            del temp_output
            del validation_purpose_loss

        # Step 2: LLM 权重更新
        optimizer_llm.zero_grad()

        current_lr = optimizer_llm.param_groups[0]['lr']
        llm_loss, target_loss, gl_loss = target_llm_step(
            llm_model=target_llm, 
            input_ids=text_input["input_ids"], 
            labels=text_input["labels"],
            masks=masks, 
            attn_mask=text_input["attention_mask"], 
            epoch=epoch, 
            args=args, 
            gl_module=grouplasso_module, 
            scaler=scaler
        )
        scaler.unscale_(optimizer_llm)
        torch.nn.utils.clip_grad_norm_(target_llm.parameters(), 1.0)
        scaler.step(optimizer_llm)
        scaler.update()

        # 更新 LLM 损失到训练记录
        reduced_llm_loss = reduce_loss(llm_loss)
        reduced_target_loss = reduce_loss(target_loss)
        reduced_gl_loss = reduce_loss(gl_loss)
        llm_loss_ave.update(reduced_llm_loss.item(), text_input["input_ids"].size(0))
        target_loss_ave.update(reduced_target_loss.item(), text_input["input_ids"].size(0))
        gl_loss_ave.update(reduced_gl_loss.item(), 1)
        
        if reduced_gl_loss <= gl_loss_threshold:
            gl_loss_counter += 1
        else:
            gl_loss_counter = 0
        
        if gl_loss_counter >= gl_loss_consecutive_steps:
            terminate_training = True

        ###############################################
        ### 在 LLM 训练后进行 Group Lasso 权重投影 
        if args.tuning_method != 'lora':
            if epoch >= (args.start_epoch_control + args.control_epochs) or skip_hypernet_training:
                grouplasso_module.lam = 2000
            else:
                grouplasso_module.lam = 50

            projection_status = grouplasso_module.project_weight(
                target_llm=target_llm.module, 
                pruning_masks=masks, 
                epoch=epoch, 
                lr=current_lr
            )
            if projection_status != True:
                print("weight_projection failed, check the code.")
            gl_loss = grouplasso_module(target_llm = target_llm.module, pruning_masks = masks, epoch=epoch)
            print(f"group lasso loss after projection: {gl_loss}")
        ###############################################
        
        # Step 3: 打印训练日志（仅限主进程）
        if i % args.log_interval == 0:
            if torch.distributed.get_rank() == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Time: {elapsed_time:.2f}s | "
                    f"Step: {i} | "
                    f"LLM Loss: {llm_loss_ave.avg:.4f} | "
                    f"Target Loss: {target_loss_ave.avg:.4f} | "
                    f"Group Lasso Loss: {gl_loss_ave.avg:.4f} | "
                    f"Hypernet Loss: {hypernet_loss_ave.avg:.4f} | "
                    f"Valid Loss: {valid_loss_ave.avg:.4f} | "
                    f"Ratio Loss: {ratio_loss_ave.avg:.4f} | "
                    f"Alignment Loss: {alignment_loss_ave.avg:.4f}"
                )

                if epoch < (args.start_epoch_control + args.control_epochs):
                    random_layers = random.sample(range(32), 5)
                    for layer_idx in random_layers:
                        layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in masks]
                        mlp_up_mask = layer_wise_masks[-1]
                        print(f"layer_{layer_idx}_mlp_up_mask_shape: {mlp_up_mask.size()}")
                        mlp_up_mask_ratio = (1 - mlp_up_mask).sum() / mlp_up_mask.numel()
                        print(f"layer_{layer_idx}_mlp_up_mask_ratio: {mlp_up_mask_ratio}")

                    print(f"Current PruningRatioLoss: {reduced_ratio_loss}")
                    print(f"Current AlignmentLoss: {reduced_align_loss}")

                start_time = time.time()

        if terminate_training == True:
            print(f"The GroupLasso Loss has been smaller than {gl_loss_threshold} for {gl_loss_consecutive_steps} steps, the training would be terminated rightnow.")
            break

    # 打印 Epoch 结束时的总结（仅限主进程）
    if torch.distributed.get_rank() == 0:
        print(f"\n===== End of Epoch {epoch} =====")
        print(
            f"Epoch {epoch} Summary:\n"
            f"LLM Loss (Avg): {llm_loss_ave.avg:.4f} | "
            f"Target Loss (Avg): {target_loss_ave.avg:.4f} | "
            f"Group Lasso Loss (Avg): {gl_loss_ave.avg:.4f} | "
            f"Hypernet Loss (Avg): {hypernet_loss_ave.avg:.4f} | "
            f"Ratio Loss (Avg): {ratio_loss_ave.avg:.4f} | "
            f"Alignment Loss (Avg): {alignment_loss_ave.avg:.4f}\n"
        )

    return return_mask, skip_hypernet_training, terminate_training

    '''
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
                hyper_loss, valid_loss, ratio_loss, alignment_loss = hypernet_step(hypernet=hyper_net, llm_model=target_llm, val_ids=val_inputs["input_ids"], attn_mask=val_inputs["attention_mask"],pruning_ratio_target=pruning_ratio_target, num_key_value=num_key_value, pruning_contribution=pruning_contribution)
                torch.nn.utils.clip_grad_norm_(hyper_net.parameters(), 1.0)
                optimizer_hyper.step()

                # generate new masks for next iteration's group lasso optimization
                with torch.no_grad():
                    hyper_net.eval()
                    mask_vec    = hyper_net.module()  
                    return_mask = copy.deepcopy(mask_vec)
                    masks       = hyper_net.module.transform_output(mask_vec)

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
    '''

def llm_tuning_train_one_epoch(
    nlp_dataloader, 
    target_llm, 
    optimizer_llm, 
    epoch, 
    args, 
    scaler
):
    print(f"Epoch {epoch} starting.............")

    # Initialize training loss holders
    llm_loss_ave = AverageMeter()
    target_loss_ave = AverageMeter()

    # Timer to measure elapsed time
    start_time = time.time()

    for i, text_input in enumerate(nlp_dataloader):
        # Step 1: LLM weight update
        optimizer_llm.zero_grad()
        current_lr = optimizer_llm.param_groups[0]['lr']

        # Forward pass and loss computation for LLM
        llm_loss, target_loss, _ = target_llm_step(
            llm_model=target_llm, 
            input_ids=text_input["input_ids"], 
            masks=None, 
            attn_mask=text_input["attention_mask"], 
            epoch=epoch, 
            args=args, 
            gl_module=None,
            scaler=scaler
        )

        # Backward pass and optimizer step
        scaler.unscale_(optimizer_llm)
        torch.nn.utils.clip_grad_norm_(target_llm.parameters(), 1.0)
        scaler.step(optimizer_llm)
        scaler.update()

        # Update loss metrics
        reduced_llm_loss = reduce_loss(llm_loss)
        reduced_target_loss = reduce_loss(target_loss)
        llm_loss_ave.update(reduced_llm_loss.item(), text_input["input_ids"].size(0))
        target_loss_ave.update(reduced_target_loss.item(), text_input["input_ids"].size(0))

        # Step 2: Print training logs (only on main process)
        if i % args.log_interval == 0 and torch.distributed.get_rank() == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Time: {elapsed_time:.2f}s | "
                f"Step: {i} | "
                f"LLM Loss: {llm_loss_ave.avg:.4f} | "
                f"Target Loss: {target_loss_ave.avg:.4f}"
            )
            start_time = time.time()

    # Print summary at the end of the epoch (only on main process)
    if torch.distributed.get_rank() == 0:
        print(f"\n===== End of Epoch {epoch} =====")
        print(
            f"Epoch {epoch} Summary:\n"
            f"LLM Loss (Avg): {llm_loss_ave.avg:.4f} | "
            f"Target Loss (Avg): {target_loss_ave.avg:.4f}\n"
        )

    return True