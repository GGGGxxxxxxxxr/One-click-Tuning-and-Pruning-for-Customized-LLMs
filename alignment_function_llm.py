import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from math import sqrt, floor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

#-----------------------------------------------------------------#
# custom weight forward() & backward() with configurable scale
class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        ctx.grad_w = grad_w
        input_clone = input.clone()
        return input_clone.float()
    @staticmethod
    def backward(ctx, grad_out):
        grad_input = ctx.grad_w * grad_out
        return grad_input, None
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# Group_Lasso Sparsity Regularization
# modified for LinearProjection from (nn.Conv2d) in ATO
# input:
#   ** [target_model_weights]: weights for series of nn.Linear in target LLMs
#   ** [pruning_masks]:        pruning_masks generated from Hypernet() 
#   output:                    [Group_lasso_loss] on masked porpotion of weights

#  **note: nn.Linear.weight.shape: [out_dim, in_dim]!
#  **note: within in Qwen2 Attn, q \ k \ v_proj has BIAS. o_proj & MLP_linears have NO BIAS. (BIAS ought to be pruned out if existing!)

# Version2.0 update: add GroupLassoRegularization for [DISP] pruning space
class Group_Lasso_regularization(nn.Module):
    def __init__(self, args, target_llm_cfg, prunable_structure):
        super().__init__()
        self.grad_mul = args.grad_mul if args else 1
        self.lam = args.gl_lam if args else 1000
        self.p_structure = prunable_structure
        self.model       = None
        self.cfg         = target_llm_cfg
        self.num_groups  = int(self.cfg.num_attention_heads / self.cfg.num_key_value_heads)
        self.scheme      = args.pruning_method

    '''
    def forward(self, target_llm, pruning_masks):
        self.model = target_llm
        gl_list = []

        # layer_iterative GroupLasso processing
        for layer_idx in range(self.cfg.num_hidden_layers):
            # extract corrsponding LLM_DecoderLayer & Masks for this layer
            cur_layer = self.model.model.layers[layer_idx]              # CasualLM.model -> LMmodel.layer -> DecoderLayer
            layer_wise_masks = [individual_mask[layer_idx,:] for individual_mask in pruning_masks]
            m_umlp = layer_wise_masks[-1]
            m_out  = layer_wise_masks[-2]
            m_K    = layer_wise_masks[:self.cfg.num_key_value_heads]
            m_V    = layer_wise_masks[self.cfg.num_key_value_heads : 2 * self.cfg.num_key_value_heads]

            # process MLP_up_mask
            mlp_g_weight = cur_layer.mlp.gate_proj.weight
            mlp_u_weight = cur_layer.mlp.up_proj.weight
            mlp_d_weight = cur_layer.mlp.down_proj.weight
            gl_loss      = ((1 - m_umlp).unsqueeze(1) * mlp_g_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                         + ((1 - m_umlp).unsqueeze(1) * mlp_u_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                         + ((1 - m_umlp).unsqueeze(0) * mlp_d_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)

            # process attn_out_mask
            attn_out_weight = cur_layer.self_attn.o_proj.weight
            gl_loss       = ((1 - m_out).unsqueeze(1) * attn_out_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                          + ((1 - m_out).unsqueeze(0) * mlp_u_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()    \
                          + ((1 - m_out).unsqueeze(0) * mlp_g_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)

            # process attn_V_mask
            # a) concate V_split_masks into mask for original WV
            V_mask = torch.cat(m_V)
            V_mask_repeated = torch.cat([t.repeat(self.num_groups) for t in m_V])
            # b) compute gl for v_weight, v_bias, out_weight
            attn_v_weight = cur_layer.self_attn.v_proj.weight
            attn_v_bias   = cur_layer.self_attn.v_proj.bias
            
            gl_loss       = ((1 - V_mask).unsqueeze(1) * attn_v_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                          + ((1 - V_mask) * attn_v_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                          + ((1 - V_mask_repeated).unsqueeze(0) * attn_out_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)

            # process attn_K_mask (Q_mask)
            # a) concate K_split_masks into mask for original WK
            K_mask = torch.cat(m_K)
            Q_mask = torch.cat([t.repeat(self.num_groups) for t in m_K])
            attn_k_weight = cur_layer.self_attn.k_proj.weight
            attn_k_bias   = cur_layer.self_attn.k_proj.bias
            attn_q_weight = cur_layer.self_attn.q_proj.weight
            attn_q_bias   = cur_layer.self_attn.q_proj.bias
            
            gl_loss       = ((1 - K_mask).unsqueeze(1) * attn_k_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                          + ((1 - K_mask) * attn_k_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                          + ((1 - Q_mask).unsqueeze(1) * attn_q_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                          + ((1 - Q_mask) * attn_q_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                          #+ ((1 - K_mask).unsqueeze(0) * attn_v_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)
        
        # sum gl_loss
        sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)
        #return sum_loss
        return sum_loss
        '''
    
    # ** we have detected the issue when launching the training script via torch.FSDP,
    # ** where GroupLasso requires the fullparam of a certain weight on each GPU, however, the [gl_loss] would retain the whole graph,
    # ** thus CUDAmem allocation would be larger and larger
    # ** this func only works with FSDP mode, and this func only works as a [verification] of the progress 
    def forward(self, target_llm, pruning_masks):
        self.model = target_llm
        gl_list = []

        # layer_iterative GroupLasso processing
        for layer_idx in range(self.cfg.num_hidden_layers):
            # extract corrsponding LLM_DecoderLayer & Masks for this layer
            cur_layer = self.model.model.layers[layer_idx]              # CasualLM.model -> LMmodel.layer -> DecoderLayer
            # fsdp capatibility
            with FSDP.summon_full_params(cur_layer):
                layer_wise_masks = [individual_mask[layer_idx,:] for individual_mask in pruning_masks]
                m_umlp = layer_wise_masks[-1]
                m_out  = layer_wise_masks[-2]
                m_K    = layer_wise_masks[:self.cfg.num_key_value_heads]
                m_V    = layer_wise_masks[self.cfg.num_key_value_heads : 2 * self.cfg.num_key_value_heads]

                # process MLP_up_mask
                mlp_g_weight = cur_layer.mlp.gate_proj.weight
                mlp_u_weight = cur_layer.mlp.up_proj.weight
                mlp_d_weight = cur_layer.mlp.down_proj.weight

                gl_loss      = ((1 - m_umlp).unsqueeze(1) * mlp_g_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_umlp).unsqueeze(1) * mlp_u_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_umlp).unsqueeze(0) * mlp_d_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
                
                #gl_list.append(gl_loss)
                gl_list.append(torch.tensor(gl_loss.item()).cuda())
                del gl_loss

                # process attn_out_mask
                attn_out_weight = cur_layer.self_attn.o_proj.weight
                gl_loss       = ((1 - m_out).unsqueeze(1) * attn_out_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_out).unsqueeze(0) * mlp_u_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()    \
                                + ((1 - m_out).unsqueeze(0) * mlp_g_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
                
                #gl_list.append(gl_loss)
                gl_list.append(torch.tensor(gl_loss.item()).cuda())
                del gl_loss

                # process attn_V_mask
                V_mask = torch.cat(m_V)
                V_mask_repeated = torch.cat([t.repeat(self.num_groups) for t in m_V])
                attn_v_weight = cur_layer.self_attn.v_proj.weight
                
                # 如果 attention_bias 存在并且 == False，则跳过 bias 的 regularization
                if hasattr(self.cfg, "attention_bias") and self.cfg.attention_bias == False:
                    gl_loss = ((1 - V_mask).unsqueeze(1) * attn_v_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                              + ((1 - V_mask_repeated).unsqueeze(0) * attn_out_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
                else:
                    attn_v_bias   = cur_layer.self_attn.v_proj.bias
                    gl_loss       = ((1 - V_mask).unsqueeze(1) * attn_v_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                    + ((1 - V_mask) * attn_v_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                                    + ((1 - V_mask_repeated).unsqueeze(0) * attn_out_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
               
                #gl_list.append(gl_loss)
                gl_list.append(torch.tensor(gl_loss.item()).cuda())
                del gl_loss

                # process attn_K_mask (Q_mask)
                K_mask = torch.cat(m_K)
                Q_mask = torch.cat([t.repeat(self.num_groups) for t in m_K])
                attn_k_weight = cur_layer.self_attn.k_proj.weight
                attn_q_weight = cur_layer.self_attn.q_proj.weight

                if hasattr(self.cfg, "attention_bias") and self.cfg.attention_bias == False:
                    gl_loss = ((1 - K_mask).unsqueeze(1) * attn_k_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                              + ((1 - Q_mask).unsqueeze(1) * attn_q_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum()
                else:
                    attn_k_bias   = cur_layer.self_attn.k_proj.bias
                    attn_q_bias   = cur_layer.self_attn.q_proj.bias
                    gl_loss       = ((1 - K_mask).unsqueeze(1) * attn_k_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                    + ((1 - K_mask) * attn_k_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                                    + ((1 - Q_mask).unsqueeze(1) * attn_q_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                    + ((1 - Q_mask) * attn_q_bias).pow(2).add(1e-8).pow(1/2.).sum()
                    
                #gl_list.append(gl_loss)
                gl_list.append(torch.tensor(gl_loss.item()).cuda())
                del gl_loss
                
        # sum gl_loss (for value tracing only)
        #sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)
        sum_loss = sum(gl_list) / len(gl_list)
        return sum_loss      
            

    # ** The LoRA-aware GroupLasso Loss has been supported in Oct.31th update;
    # ** If we choose [LoRA] over [Full-Param], only the LoRA would have physically pruning pattern, the base model's channel would be removed after the training of the Hypernet()
    # ** this func is working with DDP support, as by defauly LoRA implementation would not call FSDP mode;
    # ** in DDP mode, such GroupLassoLoss is quite easy to compute cuz each GPU has its own copy (and the same as others) of the LoRA weights locally.
    def lora_forward(self, target_llm, pruning_masks):
        gl_list    = []

        # layer_iterative GroupLasso processing based on LoRA module
        for layer_idx in range(self.cfg.num_hidden_layers):
            # extract corrsponding LLM_DecoderLayer & Masks for this layer
            cur_layer = target_llm.model.layers[layer_idx]                                          # CasualLM.model -> LMmodel.layer -> DecoderLayer
            layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in pruning_masks]
            m_umlp = layer_wise_masks[-1]
            #m_out  = layer_wise_masks[-2]
            m_K    = layer_wise_masks[-3]
            m_V    = layer_wise_masks[-2]
            assert len(layer_wise_masks) == 3, 'check the implementation in [lora_forward]'

            # process MLP_up_mask for LoRA weights
            mlp_g_lora_B = cur_layer.mlp.gate_proj.lora_B
            mlp_u_lora_B = cur_layer.mlp.up_proj.lora_B
            mlp_d_lora_A = cur_layer.mlp.down_proj.lora_A

            gl_loss      = ((1 - m_umlp).unsqueeze(1) * mlp_g_lora_B).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_umlp).unsqueeze(1) * mlp_u_lora_B).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_umlp).unsqueeze(0) * mlp_d_lora_A).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)

            
            '''
            # process attn_out_mask
            attn_out_lora_B = cur_layer.self_attn.o_proj.lora_B
            mlp_g_lora_A    = cur_layer.mlp.gate_proj.lora_A
            mlp_u_lora_A    = cur_layer.mlp.up_proj.lora_A

            gl_loss       = ((1 - m_out).unsqueeze(1) * attn_out_lora_B).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                          + ((1 - m_out).unsqueeze(0) * mlp_g_lora_A).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()    \
                          + ((1 - m_out).unsqueeze(0) * mlp_u_lora_A).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)
            '''

            # process attn_V_mask
            # a) concate V_split_masks into mask for original WV
            V_mask = m_V
            V_mask_repeated = V_mask.repeat(self.num_groups)
            
            # b) compute gl for v_weight, v_bias, out_weight
            attn_v_lora_B   = cur_layer.self_attn.v_proj.lora_B
            attn_out_lora_A = cur_layer.self_attn.o_proj.lora_A

            gl_loss       = ((1 - V_mask).unsqueeze(1) * attn_v_lora_B).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                          + ((1 - V_mask_repeated).unsqueeze(0) * attn_out_lora_A).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)

            # process attn_K_mask (Q_mask)
            # a) concate K_split_masks into mask for original WK
            K_mask = m_K
            Q_mask = K_mask.repeat(self.num_groups)
            
            attn_k_lora_B = cur_layer.self_attn.k_proj.lora_B
            attn_q_lora_B = cur_layer.self_attn.q_proj.lora_B
            
            gl_loss       = ((1 - K_mask).unsqueeze(1) * attn_k_lora_B).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                          + ((1 - Q_mask).unsqueeze(1) * attn_q_lora_B).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                          #+ ((1 - K_mask).unsqueeze(0) * attn_v_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)
            

        # sum gl_loss
        #sum_loss = sum(gl_list)/len(gl_list)

        sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)

        #test
        '''
        sum_loss.backward()
        mlp_g_lora_B = cur_layer.mlp.gate_proj.lora_B
        print(mlp_g_lora_B.grad)
        '''
        return sum_loss






    # several implementations for GroupLasso in FSDP mode have been tried.
    # 1. direct computation of gl_loss(sum) + target_llm_loss --> CUDAmem bloaded 
    # 2. instant backward() after gl_loss computed for a certain group --> the gradient for this group would be totally ignored after context exits
    # The official doc of FSDP notes that backward() / forward() could not be called within the summon_full_params() context!!
    # thus, for FSDP mode, the direct weight_projection is expected if your model is too large to be fit within DDP mode
    # the group_lasso_proximal solution would be directly applied to the target group of weight.
    def project_weight(self, target_llm, pruning_masks, epoch, lr):
        self.model = target_llm

        # Calculate ratio
        N_t = 0
        for msk in pruning_masks:
            N_t += (1 - msk).sum()

        with torch.no_grad():  # Ensure no gradients are recorded
            # Iterate over each layer for Group Lasso processing
            for layer_idx in range(self.cfg.num_hidden_layers):
                # Extract corresponding LLM decoder layer & masks
                cur_layer = self.model.model.layers[layer_idx]

                with FSDP.summon_full_params(cur_layer):
                    # Extract masks for the current layer
                    layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in pruning_masks]
                    m_umlp = layer_wise_masks[-1]
                    m_out = layer_wise_masks[-2]
                    m_K = layer_wise_masks[:self.cfg.num_key_value_heads]
                    m_V = layer_wise_masks[self.cfg.num_key_value_heads: 2 * self.cfg.num_key_value_heads]

                    # Process MLP mask (Gate/Up/DownProj)
                    ratio = (1 - m_umlp).sum() / N_t
                    if ratio > 0:
                        mlp_g_weight = cur_layer.mlp.gate_proj.weight.data
                        mlp_u_weight = cur_layer.mlp.up_proj.weight.data
                        mlp_d_weight = cur_layer.mlp.down_proj.weight.data

                        m_umlp_out = (m_umlp == 0)
                        w_norm = mlp_g_weight[m_umlp_out, :].pow(2).sum(1) + \
                                mlp_u_weight[m_umlp_out, :].pow(2).sum(1) + \
                                mlp_d_weight[:, m_umlp_out].pow(2).sum(0)
                        w_norm = w_norm.add(1e-8).pow(0.5)

                        mlp_g_weight[m_umlp_out, :] /= w_norm.unsqueeze(1)
                        mlp_u_weight[m_umlp_out, :] /= w_norm.unsqueeze(1)
                        mlp_d_weight[:, m_umlp_out] /= w_norm.unsqueeze(0)

                        tmp = (-self.lam * lr + w_norm).clamp(min=0) #* 0  # Apply mask scaling factor

                        mlp_g_weight[m_umlp_out, :] *= tmp.unsqueeze(1)
                        mlp_u_weight[m_umlp_out, :] *= tmp.unsqueeze(1)
                        mlp_d_weight[:, m_umlp_out] *= tmp.unsqueeze(0)

                        cur_layer.mlp.gate_proj.weight.copy_(mlp_g_weight)
                        cur_layer.mlp.up_proj.weight.copy_(mlp_u_weight)
                        cur_layer.mlp.down_proj.weight.copy_(mlp_d_weight)

                    '''
                    test purpose
                    cur_layer.mlp.down_proj.weight.zero_()
                    '''    
                        
                    # Process attention out mask
                    ratio = (1 - m_out).sum() / N_t
                    if ratio > 0:
                        attn_out_weight = cur_layer.self_attn.o_proj.weight.data
                        mlp_g_weight = cur_layer.mlp.gate_proj.weight.data
                        mlp_u_weight = cur_layer.mlp.up_proj.weight.data

                        m_out = (m_out == 0)
                        w_norm = mlp_g_weight[:, m_out].pow(2).sum(0) + \
                                mlp_u_weight[:, m_out].pow(2).sum(0) + \
                                attn_out_weight[m_out, :].pow(2).sum(1)
                        w_norm = w_norm.add(1e-8).pow(0.5)

                        mlp_g_weight[:, m_out] /= w_norm.unsqueeze(0)
                        mlp_u_weight[:, m_out] /= w_norm.unsqueeze(0)
                        attn_out_weight[m_out, :] /= w_norm.unsqueeze(1)

                        tmp = (-self.lam * lr + w_norm).clamp(min=0) #* 0

                        mlp_g_weight[:, m_out] *= tmp.unsqueeze(0)
                        mlp_u_weight[:, m_out] *= tmp.unsqueeze(0)
                        attn_out_weight[m_out, :] *= tmp.unsqueeze(1)

                        cur_layer.mlp.gate_proj.weight.copy_(mlp_g_weight)
                        cur_layer.mlp.up_proj.weight.copy_(mlp_u_weight)
                        cur_layer.self_attn.o_proj.weight.copy_(attn_out_weight)

                    # Process V mask
                    V_mask = torch.cat(m_V)
                    V_mask_repeated = torch.cat([t.repeat(self.num_groups) for t in m_V])
                    ratio = (1 - V_mask).sum() / N_t

                    if ratio > 0:
                        V_mask = (V_mask == 0)
                        V_mask_repeated = (V_mask_repeated == 0)

                        attn_v_weight = cur_layer.self_attn.v_proj.weight.data
                        attn_out_weight = cur_layer.self_attn.o_proj.weight.data

                        w_norm = attn_v_weight[V_mask, :].pow(2).sum(1) + \
                                attn_out_weight[:, V_mask_repeated].pow(2).sum(0)
                        w_norm = w_norm.add(1e-8).pow(0.5)

                        attn_v_weight[V_mask, :] /= w_norm.unsqueeze(1)
                        attn_out_weight[:, V_mask_repeated] /= w_norm.unsqueeze(0)

                        tmp = (-self.lam * lr + w_norm).clamp(min=0) #* 0

                        attn_v_weight[V_mask, :] *= tmp.unsqueeze(1)
                        attn_out_weight[:, V_mask_repeated] *= tmp.unsqueeze(0)

                        cur_layer.self_attn.v_proj.weight.copy_(attn_v_weight)
                        cur_layer.self_attn.o_proj.weight.copy_(attn_out_weight)

                    # Process K mask (Q mask)
                    K_mask = torch.cat(m_K)
                    Q_mask = torch.cat([t.repeat(self.num_groups) for t in m_K])
                    ratio = (1 - K_mask).sum() / N_t

                    if ratio > 0:
                        m_K_out = (K_mask == 0)
                        m_Q_out = (Q_mask == 0)

                        attn_k_weight = cur_layer.self_attn.k_proj.weight.data
                        attn_q_weight = cur_layer.self_attn.q_proj.weight.data

                        w_norm = attn_k_weight[m_K_out, :].pow(2).sum(1) + \
                                attn_q_weight[m_Q_out, :].pow(2).sum(1)
                        w_norm = w_norm.add(1e-8).pow(0.5)

                        attn_k_weight[m_K_out, :] /= w_norm.unsqueeze(1)
                        attn_q_weight[m_Q_out, :] /= w_norm.unsqueeze(1)

                        tmp = (-self.lam * lr + w_norm).clamp(min=0) #* 0

                        attn_k_weight[m_K_out, :] *= tmp.unsqueeze(1)
                        attn_q_weight[m_Q_out, :] *= tmp.unsqueeze(1)

                        cur_layer.self_attn.k_proj.weight.copy_(attn_k_weight)
                        cur_layer.self_attn.q_proj.weight.copy_(attn_q_weight)

                    '''
                    dist.barrier()
                    for param in cur_layer.parameters():
                        dist.broadcast(param, src=0)
                    dist.barrier()
                    '''
                '''
                with FSDP.summon_full_params(cur_layer):
                    if torch.all(cur_layer.mlp.down_proj.weight == 0):
                        print("All down_proj weights are correctly set to zero.")
                    else:
                        print("Error: Some down_proj weights are not zero!")
                '''
            
        return True
    


    '''
    the following functions are all for debugging
    '''
    def debug_purpose_compute(self, target_llm, pruning_masks, epoch):
        self.model = target_llm
        gl_list = []

        # layer_iterative GroupLasso processing
        for layer_idx in range(self.cfg.num_hidden_layers):

            print(f"cur_layer: {layer_idx}")

            # extract corrsponding LLM_DecoderLayer & Masks for this layer
            cur_layer = self.model.model.layers[layer_idx]              # CasualLM.model -> LMmodel.layer -> DecoderLayer
            
            layer_wise_masks = [individual_mask[layer_idx,:] for individual_mask in pruning_masks]
            m_umlp = layer_wise_masks[-1]
            m_out  = layer_wise_masks[-2]
            m_K    = layer_wise_masks[:self.cfg.num_key_value_heads]
            m_V    = layer_wise_masks[self.cfg.num_key_value_heads : 2 * self.cfg.num_key_value_heads]

            # process MLP_up_mask
            mlp_g_weight = cur_layer.mlp.gate_proj.weight
            mlp_u_weight = cur_layer.mlp.up_proj.weight
            mlp_d_weight = cur_layer.mlp.down_proj.weight
            
            gl_loss      = ((1 - m_umlp).unsqueeze(1) * mlp_g_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                            + ((1 - m_umlp).unsqueeze(1) * mlp_u_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                            + ((1 - m_umlp).unsqueeze(0) * mlp_d_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            
            print(f"gl_loss_for_mlp_up_mask:{gl_loss}")

            gl_list.append(torch.tensor(gl_loss.item()).cuda())
            del gl_loss

            # process attn_out_mask
            attn_out_weight = cur_layer.self_attn.o_proj.weight
            gl_loss       = ((1 - m_out).unsqueeze(1) * attn_out_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                            + ((1 - m_out).unsqueeze(0) * mlp_u_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()    \
                            + ((1 - m_out).unsqueeze(0) * mlp_g_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            
            print(f"gl_loss_for_attn_out_mask:{gl_loss}")

            gl_list.append(torch.tensor(gl_loss.item()).cuda())
            del gl_loss

            # process attn_V_mask
            V_mask = torch.cat(m_V)
            V_mask_repeated = torch.cat([t.repeat(self.num_groups) for t in m_V])
            attn_v_weight = cur_layer.self_attn.v_proj.weight
            
            # 如果 attention_bias 存在并且 == False，则跳过 bias 的 regularization
            if hasattr(self.cfg, "attention_bias") and self.cfg.attention_bias == False:
                gl_loss = ((1 - V_mask).unsqueeze(1) * attn_v_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                            + ((1 - V_mask_repeated).unsqueeze(0) * attn_out_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            else:
                attn_v_bias   = cur_layer.self_attn.v_proj.bias
                gl_loss       = ((1 - V_mask).unsqueeze(1) * attn_v_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - V_mask) * attn_v_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                                + ((1 - V_mask_repeated).unsqueeze(0) * attn_out_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            
            print(f"gl_loss_for_attn_v_mask:{gl_loss}")

            gl_list.append(torch.tensor(gl_loss.item()).cuda())
            del gl_loss

            # process attn_K_mask (Q_mask)
            K_mask = torch.cat(m_K)
            Q_mask = torch.cat([t.repeat(self.num_groups) for t in m_K])
            attn_k_weight = cur_layer.self_attn.k_proj.weight
            attn_q_weight = cur_layer.self_attn.q_proj.weight

            if hasattr(self.cfg, "attention_bias") and self.cfg.attention_bias == False:
                gl_loss = ((1 - K_mask).unsqueeze(1) * attn_k_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                            + ((1 - Q_mask).unsqueeze(1) * attn_q_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum()
            else:
                attn_k_bias   = cur_layer.self_attn.k_proj.bias
                attn_q_bias   = cur_layer.self_attn.q_proj.bias
                gl_loss       = ((1 - K_mask).unsqueeze(1) * attn_k_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - K_mask) * attn_k_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                                + ((1 - Q_mask).unsqueeze(1) * attn_q_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - Q_mask) * attn_q_bias).pow(2).add(1e-8).pow(1/2.).sum()
                
            print(f"gl_loss_for_attn_k_mask:{gl_loss}")

            gl_list.append(torch.tensor(gl_loss.item()).cuda())
            del gl_loss
        
        # sum gl_loss (for value tracing only)
        #sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)
        sum_loss = sum(gl_list) / len(gl_list)
        print(f"group_lasso_loss: {sum_loss}")
        return sum_loss    
    


    def debug_purpose_nofsdp_project_weight(self, target_llm, pruning_masks, epoch, lr):
        self.model = target_llm

        # Adjust regularization intensity
        if epoch >= 20:
            self.lam = 200000000

        # Calculate ratio
        N_t = 0
        for msk in pruning_masks:
            N_t += (1 - msk).sum()

        with torch.no_grad():  # Ensure no gradients are recorded
            # Iterate over each layer for Group Lasso processing
            for layer_idx in range(self.cfg.num_hidden_layers):
                # Extract corresponding LLM decoder layer & masks
                cur_layer = self.model.model.layers[layer_idx]

                # Extract masks for the current layer
                layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in pruning_masks]
                m_umlp = layer_wise_masks[-1]
                m_out = layer_wise_masks[-2]
                m_K = layer_wise_masks[:self.cfg.num_key_value_heads]
                m_V = layer_wise_masks[self.cfg.num_key_value_heads: 2 * self.cfg.num_key_value_heads]

                # Process MLP mask (Gate/Up/DownProj)
                ratio = (1 - m_umlp).sum() / N_t
                if ratio > 0:
                    mlp_g_weight = cur_layer.mlp.gate_proj.weight.data
                    mlp_u_weight = cur_layer.mlp.up_proj.weight.data
                    mlp_d_weight = cur_layer.mlp.down_proj.weight.data

                    m_umlp_out = (m_umlp == 0)
                    w_norm = mlp_g_weight[m_umlp_out, :].pow(2).sum(1) + \
                            mlp_u_weight[m_umlp_out, :].pow(2).sum(1) + \
                            mlp_d_weight[:, m_umlp_out].pow(2).sum(0)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    mlp_g_weight[m_umlp_out, :] /= w_norm.unsqueeze(1)
                    mlp_u_weight[m_umlp_out, :] /= w_norm.unsqueeze(1)
                    mlp_d_weight[:, m_umlp_out] /= w_norm.unsqueeze(0)

                    tmp = (-self.lam * lr + w_norm).clamp(min=0) * 0  # Apply mask scaling factor

                    mlp_g_weight[m_umlp_out, :] *= tmp.unsqueeze(1)
                    mlp_u_weight[m_umlp_out, :] *= tmp.unsqueeze(1)
                    mlp_d_weight[:, m_umlp_out] *= tmp.unsqueeze(0)

                    cur_layer.mlp.gate_proj.weight.copy_(mlp_g_weight)
                    cur_layer.mlp.up_proj.weight.copy_(mlp_u_weight)
                    cur_layer.mlp.down_proj.weight.copy_(mlp_d_weight)

                # Process attention out mask
                ratio = (1 - m_out).sum() / N_t
                if ratio > 0:
                    attn_out_weight = cur_layer.self_attn.o_proj.weight.data
                    mlp_g_weight = cur_layer.mlp.gate_proj.weight.data
                    mlp_u_weight = cur_layer.mlp.up_proj.weight.data

                    m_out = (m_out == 0)
                    w_norm = mlp_g_weight[:, m_out].pow(2).sum(0) + \
                            mlp_u_weight[:, m_out].pow(2).sum(0) + \
                            attn_out_weight[m_out, :].pow(2).sum(1)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    mlp_g_weight[:, m_out] /= w_norm.unsqueeze(0)
                    mlp_u_weight[:, m_out] /= w_norm.unsqueeze(0)
                    attn_out_weight[m_out, :] /= w_norm.unsqueeze(1)

                    tmp = (-self.lam * lr + w_norm).clamp(min=0) * 0

                    mlp_g_weight[:, m_out] *= tmp.unsqueeze(0)
                    mlp_u_weight[:, m_out] *= tmp.unsqueeze(0)
                    attn_out_weight[m_out, :] *= tmp.unsqueeze(1)

                    cur_layer.mlp.gate_proj.weight.copy_(mlp_g_weight)
                    cur_layer.mlp.up_proj.weight.copy_(mlp_u_weight)
                    cur_layer.self_attn.o_proj.weight.copy_(attn_out_weight)

                # Process V mask
                V_mask = torch.cat(m_V)
                V_mask_repeated = torch.cat([t.repeat(self.num_groups) for t in m_V])
                ratio = (1 - V_mask).sum() / N_t

                if ratio > 0:
                    V_mask = (V_mask == 0)
                    V_mask_repeated = (V_mask_repeated == 0)

                    attn_v_weight = cur_layer.self_attn.v_proj.weight.data
                    attn_out_weight = cur_layer.self_attn.o_proj.weight.data

                    w_norm = attn_v_weight[V_mask, :].pow(2).sum(1) + \
                            attn_out_weight[:, V_mask_repeated].pow(2).sum(0)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    attn_v_weight[V_mask, :] /= w_norm.unsqueeze(1)
                    attn_out_weight[:, V_mask_repeated] /= w_norm.unsqueeze(0)

                    tmp = (-self.lam * lr + w_norm).clamp(min=0) * 0

                    attn_v_weight[V_mask, :] *= tmp.unsqueeze(1)
                    attn_out_weight[:, V_mask_repeated] *= tmp.unsqueeze(0)

                    cur_layer.self_attn.v_proj.weight.copy_(attn_v_weight)
                    cur_layer.self_attn.o_proj.weight.copy_(attn_out_weight)

                # Process K mask (Q mask)
                K_mask = torch.cat(m_K)
                Q_mask = torch.cat([t.repeat(self.num_groups) for t in m_K])
                ratio = (1 - K_mask).sum() / N_t

                if ratio > 0:
                    m_K_out = (K_mask == 0)
                    m_Q_out = (Q_mask == 0)

                    attn_k_weight = cur_layer.self_attn.k_proj.weight.data
                    attn_q_weight = cur_layer.self_attn.q_proj.weight.data

                    w_norm = attn_k_weight[m_K_out, :].pow(2).sum(1) + \
                            attn_q_weight[m_Q_out, :].pow(2).sum(1)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    attn_k_weight[m_K_out, :] /= w_norm.unsqueeze(1)
                    attn_q_weight[m_Q_out, :] /= w_norm.unsqueeze(1)

                    tmp = (-self.lam * lr + w_norm).clamp(min=0) * 0

                    attn_k_weight[m_K_out, :] *= tmp.unsqueeze(1)
                    attn_q_weight[m_Q_out, :] *= tmp.unsqueeze(1)

                    cur_layer.self_attn.k_proj.weight.copy_(attn_k_weight)
                    cur_layer.self_attn.q_proj.weight.copy_(attn_q_weight)
                
        return True


# [ATP_DISP]: optimized GroupLasso implementation tailored for DISP pruning space
class Group_Lasso_regularization_DISP(nn.Module):
    def __init__(self, args, target_llm_cfg, prunable_structure):
        super().__init__()
        self.grad_mul = args.grad_mul if args else 1
        self.lam = args.gl_lam if args else 1000
        self.p_structure = prunable_structure
        self.model       = None
        self.cfg         = target_llm_cfg
        self.num_groups  = int(self.cfg.num_attention_heads / self.cfg.num_key_value_heads)
        self.scheme      = args.pruning_method
        self.lr          = None

    ## Version2.0 updated: groupproximal + group_lasso_loss_tracker for [DISP] pruning space
    ## **notice: we recommend to directly perform GroupLasso projection after weight update, and using gl_loss periodially just for sparsity convergenecy tracker
    def project_weight_lora_DISP(self, target_llm, pruning_masks):
        self.model = target_llm

        # Calculate ratio
        ## **notice: the ratio is utilized to approximate the different contributions of corresponding rows / columns,
        ## those groups with relatively larger overall impact would be penalized through approximal more hard;
        N_t = 0
        for msk in pruning_masks:
            N_t += (1 - msk).sum()

        with torch.no_grad():                                                         # Ensure no gradients are recorded
            # Iterate over each layer for GroupLassoProximal
            for layer_idx in range(self.cfg.num_hidden_layers):
                # Extract corresponding LLM decoder layer & masks
                cur_layer = self.model.model.layers[layer_idx]

                # Extract s1-s5 for the current layer
                layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in pruning_masks]
                m_s1 = layer_wise_masks[0]
                m_s2 = layer_wise_masks[1]
                m_s3 = layer_wise_masks[2]
                m_s4 = layer_wise_masks[3]
                m_s5 = layer_wise_masks[4]

                # [ATP_DISP]: 1. process s1
                ratio = (1 - m_s1).sum() / N_t

                if ratio > 0:
                    # acquire current weight tensors
                    attn_q_lA = cur_layer.self_attn.q_proj.lora_A
                    attn_k_lA = cur_layer.self_attn.k_proj.lora_A
                    attn_v_lA = cur_layer.self_attn.v_proj.lora_A

                    # calculate grouped w_norm
                    m_s1 = (m_s1 == 0)
                    w_norm = attn_q_lA[m_s1, :].pow(2).sum(1) + \
                             attn_k_lA[m_s1, :].pow(2).sum(1) + \
                             attn_v_lA[m_s1, :].pow(2).sum(1)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    attn_q_lA.copy_(self.groupproximal(attn_q_lA, m_s1, ratio, w_norm, 'in_dim'))
                    attn_k_lA.copy_(self.groupproximal(attn_k_lA, m_s1, ratio, w_norm, 'in_dim'))
                    attn_v_lA.copy_(self.groupproximal(attn_v_lA, m_s1, ratio, w_norm, 'in_dim'))
                    
                # [ATP_DISP]: 2. process s2
                ratio = (1 - m_s2).sum() / N_t
                if ratio > 0:
                    attn_o_lB = cur_layer.self_attn.o_proj.lora_B

                    m_s2 = (m_s2 == 0)
                    w_norm = attn_o_lB[:, m_s2].pow(2).sum(0)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    attn_o_lB.copy_(self.groupproximal(attn_o_lB, m_s2, ratio, w_norm, 'out_dim'))

                # [ATP_DISP]: 3. process s3
                ratio = (1 - m_s3).sum() / N_t
                if ratio > 0:
                    m_s3 = (m_s3 == 0)
            
                    mlp_u_lA = cur_layer.mlp.up_proj.lora_A
                    mlp_g_lA = cur_layer.mlp.gate_proj.lora_A

                    w_norm = mlp_u_lA[m_s3, :].pow(2).sum(1) + \
                             mlp_g_lA[m_s3, :].pow(2).sum(1)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    mlp_u_lA.copy_(self.groupproximal(mlp_u_lA, m_s3, ratio, w_norm, 'in_dim'))
                    mlp_g_lA.copy_(self.groupproximal(mlp_g_lA, m_s3, ratio, w_norm, 'in_dim'))

                # [ATP_DISP]: 4. process s4
                ratio = (1 - m_s4).sum() / N_t
                if ratio > 0:
                    m_s4 = (m_s4 == 0)
            
                    mlp_u_lB = cur_layer.mlp.up_proj.lora_B
                    mlp_g_lB = cur_layer.mlp.gate_proj.lora_B 
                    mlp_d_lA = cur_layer.mlp.down_proj.lora_A
                    w_norm = mlp_u_lB[:, m_s4].pow(2).sum(0) + \
                             mlp_g_lB[:, m_s4].pow(2).sum(0) + \
                             mlp_d_lA[m_s4, :].pow(2).sum(1)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    mlp_u_lB.copy_(self.groupproximal(mlp_u_lB, m_s4, ratio, w_norm, 'out_dim'))
                    mlp_g_lB.copy_(self.groupproximal(mlp_g_lB, m_s4, ratio, w_norm, 'out_dim'))
                    mlp_d_lA.copy_(self.groupproximal(mlp_d_lA, m_s4, ratio, w_norm, 'in_dim'))
                
                # [ATP_DISP]: 5. process s5
                ratio = (1 - m_s5).sum() / N_t
                if ratio > 0:
                    m_s5 = (m_s5 == 0)
            
                    mlp_d_lB = cur_layer.mlp.down_proj.lora_B

                    w_norm = mlp_d_lB[:, m_s5].pow(2).sum(0)
                    w_norm = w_norm.add(1e-8).pow(0.5)

                    mlp_d_lB.copy_(self.groupproximal(mlp_d_lB, m_s5, ratio, w_norm, 'out_dim'))
            
        return True


    # group_lasso_proximal_solution for a single matrics
    def groupproximal(self, weight, m_s, ratio, w_norm, dim):
        if dim == 'in_dim':
            w_norm = w_norm.unsqueeze(1)
            weight[m_s, :]   = weight[m_s, :] / w_norm
            scale            = - self.grad_mul * ratio * self.lr + w_norm
            scale[scale < 0] = 0
            weight[m_s, :]   = weight[m_s, :] * scale
        else:
            weight[:, m_s]   = weight[:, m_s] / w_norm
            scale            = - self.grad_mul * ratio * self.lr + w_norm
            scale[scale < 0] = 0
            weight[:, m_s]   = weight[:, m_s] * scale
        
        return weight

    # for current structural sparsity degree tracking purpose
    # ** in version2.0, we dont use it for training
    def lora_DISP_forward(self, target_llm, pruning_masks):
        self.model = target_llm
        gl_list    = []

        with torch.no_grad():                                                         # Ensure no gradients are recorded
            # Iterate over each layer for GroupLassoProximal
            for layer_idx in range(self.cfg.num_hidden_layers):
                # Extract corresponding LLM decoder layer & masks
                cur_layer = self.model.model.layers[layer_idx]

                # Extract s1-s5 for the current layer
                layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in pruning_masks]
                m_s1 = layer_wise_masks[0]
                m_s2 = layer_wise_masks[1]
                m_s3 = layer_wise_masks[2]
                m_s4 = layer_wise_masks[3]
                m_s5 = layer_wise_masks[4]

                # [ATP_DISP]: 1. process s1
                # acquire current weight tensors
                attn_q_lA = cur_layer.self_attn.q_proj.lora_A
                attn_k_lA = cur_layer.self_attn.k_proj.lora_A
                attn_v_lA = cur_layer.self_attn.v_proj.lora_A

                # calculate grouped w_norm
                m_s1 = (m_s1 == 0)
                w_norm = attn_q_lA[m_s1, :].pow(2).sum(1) + \
                            attn_k_lA[m_s1, :].pow(2).sum(1) + \
                            attn_v_lA[m_s1, :].pow(2).sum(1)
                w_norm = w_norm.add(1e-8).pow(0.5).sum()
                gl_list.append(w_norm)
                    
                # [ATP_DISP]: 2. process s2
                attn_o_lB = cur_layer.self_attn.o_proj.lora_B

                m_s2 = (m_s2 == 0)
                w_norm = attn_o_lB[:, m_s2].pow(2).sum(0)
                w_norm = w_norm.add(1e-8).pow(0.5).sum()
                gl_list.append(w_norm)

                # [ATP_DISP]: 3. process s3
                m_s3 = (m_s3 == 0)
        
                mlp_u_lA = cur_layer.mlp.up_proj.lora_A
                mlp_g_lA = cur_layer.mlp.gate_proj.lora_A

                w_norm = mlp_u_lA[m_s3, :].pow(2).sum(1) + \
                            mlp_g_lA[m_s3, :].pow(2).sum(1)
                w_norm = w_norm.add(1e-8).pow(0.5).sum()
                gl_list.append(w_norm)

                # [ATP_DISP]: 4. process s4
                m_s4 = (m_s4 == 0)
        
                mlp_u_lB = cur_layer.mlp.up_proj.lora_B
                mlp_g_lB = cur_layer.mlp.gate_proj.lora_B 
                mlp_d_lA = cur_layer.mlp.down_proj.lora_A
                w_norm = mlp_u_lB[:, m_s4].pow(2).sum(0) + \
                            mlp_g_lB[:, m_s4].pow(2).sum(0) + \
                            mlp_d_lA[m_s4, :].pow(2).sum(1)
                w_norm = w_norm.add(1e-8).pow(0.5).sum()
                gl_list.append(w_norm)
                
                # [ATP_DISP]: 5. process s5
                m_s5 = (m_s5 == 0)
    
                mlp_d_lB = cur_layer.mlp.down_proj.lora_B

                w_norm = mlp_d_lB[:, m_s5].pow(2).sum(0)
                w_norm = w_norm.add(1e-8).pow(0.5).sum()
                gl_list.append(w_norm)

        # return averaged current structural sparsity degree
        ave_gl = sum(gl_list) / len(gl_list)
        return ave_gl






        






