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
class Group_Lasso_regularization(nn.Module):
    def __init__(self, args, target_llm_cfg, prunable_structure, fsdp_scaler):
        super().__init__()
        self.grad_mul    = 1 #args.grad_mul
        self.lam         = 10000 #args.gl_lam
        self.p_structure = prunable_structure
        self.model       = None
        self.cfg         = target_llm_cfg
        self.num_groups  = int(self.cfg.num_attention_heads / self.cfg.num_key_value_heads)
        self.scaler      = fsdp_scaler

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
    # we have detected the issue when launching the training script via torch.FSDP,
    # where GroupLasso requires the fullparam of a certain weight on each GPU, however, the [gl_loss] would retain the whole graph,
    # thus CUDAmem allocation would be larger and larger
    def forward(self, target_llm, pruning_masks, epoch):
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
                
                gl_list.append(torch.tensor(gl_loss.item()).cuda())
                del gl_loss

                # process attn_out_mask
                attn_out_weight = cur_layer.self_attn.o_proj.weight
                gl_loss       = ((1 - m_out).unsqueeze(1) * attn_out_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_out).unsqueeze(0) * mlp_u_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()    \
                                + ((1 - m_out).unsqueeze(0) * mlp_g_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
                
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
                    
                
                gl_list.append(torch.tensor(gl_loss.item()).cuda())
                del gl_loss
                
        # sum gl_loss (for value tracing only)
        #sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)
        sum_loss = sum(gl_list) / len(gl_list)
        return sum_loss              

    
    # several implementations for GroupLasso in FSDP mode have been tried.
    # 1. direct computation of gl_loss(sum) + target_llm_loss --> CUDAmem bloaded 
    # 2. instant backward() after gl_loss computed for a certain group --> the gradient for this group would be totally ignored after context exits
    # The official doc of FSDP notes that backward() / forward() could not be called within the summon_full_params() context!!
    # thus, for FSDP mode, the direct weight_projection is expected if your model is too large to be fit within DDP mode
    # the group_lasso_proximal solution would be directly applied to the target group of weight.
    def project_weight(self, target_llm, pruning_masks, epoch, lr):
        self.model = target_llm

        # adjust regularization tensity
        if epoch >= 20:
            self.lam = 200000000

        '''
        if epoch >= 10:
            self.lam = 100 * self.lam
        '''

        # ratio
        N_t = 0
        for msk in pruning_masks:
            N_t += (1-msk).sum()
        
        with torch.no_grad():
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

                    # process MLP_up_mask (weight_projection for Gate/Up/DownProj)
                    ratio = (1 - m_umlp).sum() / N_t

                    if ratio > 0:
                        mlp_g_weight = cur_layer.mlp.gate_proj.weight.data
                        mlp_u_weight = cur_layer.mlp.up_proj.weight.data
                        mlp_d_weight = cur_layer.mlp.down_proj.weight.data

                        m_umlp_out = (m_umlp == 0)
                        w_norm =  mlp_g_weight[m_umlp_out,:].pow(2).sum((1))
                        w_norm += mlp_u_weight[m_umlp_out,:].pow(2).sum((1))
                        w_norm += mlp_d_weight[:,m_umlp_out].pow(2).sum((0))
                        w_norm = w_norm.add(1e-8).pow(1/2.)

                        mlp_g_weight[m_umlp_out,:] = mlp_g_weight[m_umlp_out,:] / w_norm.unsqueeze(1)
                        mlp_u_weight[m_umlp_out,:] = mlp_u_weight[m_umlp_out,:] / w_norm.unsqueeze(1)
                        mlp_d_weight[:,m_umlp_out] = mlp_d_weight[:,m_umlp_out] / w_norm.unsqueeze(0)

                        tmp = - self.lam * lr + w_norm #* ratio + w_norm
                        tmp[tmp<0] = 0

                        tmp *= 0

                        mlp_g_weight[m_umlp_out,:] = mlp_g_weight[m_umlp_out,:] * tmp.unsqueeze(1)
                        mlp_u_weight[m_umlp_out,:] = mlp_u_weight[m_umlp_out,:] * tmp.unsqueeze(1)
                        mlp_d_weight[:,m_umlp_out] = mlp_d_weight[:,m_umlp_out] * tmp.unsqueeze(0)

                        cur_layer.mlp.gate_proj.weight.copy_(mlp_g_weight)
                        cur_layer.mlp.up_proj.weight.copy_(mlp_u_weight)
                        cur_layer.mlp.down_proj.weight.copy_(mlp_d_weight)

                        "DEBUG PURPOSE print(w_norm and tmp for Layer0)"
                        #if layer_idx == 0:
                            #print(f"current mask pattern: {w_norm.size()}")
                            #print(f"w_norm_for_mlpU_layer{layer_idx}: {w_norm}")
                            #print(f"tmp_for_mlpU_layer{layer_idx}:    {tmp}\n")
                        '''
                        ** test for weight_copy within FSDP.summon_full_params()
                        down_proj_size = cur_layer.mlp.down_proj.weight.size()
                        zero_weight = torch.zeros(down_proj_size).to(cur_layer.mlp.down_proj.weight.device)
                        cur_layer.mlp.down_proj.weight.copy_(zero_weight)
                        '''

                    # process attn_out_mask （weight_projection for attn_out_mask)
                    ratio = (1 - m_out).sum() / N_t

                    if ratio > 0:
                        attn_out_weight = cur_layer.self_attn.o_proj.weight.data
                        mlp_g_weight = cur_layer.mlp.gate_proj.weight.data
                        mlp_u_weight = cur_layer.mlp.up_proj.weight.data

                        m_out = (m_out == 0)
                        w_norm =  mlp_g_weight[:,m_out].pow(2).sum((0))
                        w_norm += mlp_u_weight[:,m_out].pow(2).sum((0))
                        w_norm += attn_out_weight[m_out, :].pow(2).sum((1))
                        w_norm =  w_norm.add(1e-8).pow(1/2.)

                        mlp_g_weight[:,m_out] = mlp_g_weight[:,m_out] / w_norm.unsqueeze(0)
                        mlp_u_weight[:,m_out] = mlp_u_weight[:,m_out] / w_norm.unsqueeze(0)
                        attn_out_weight[m_out,:] = attn_out_weight[m_out,:] / w_norm.unsqueeze(1)

                        tmp = -self.lam * lr + w_norm #* ratio + w_norm
                        tmp[tmp<0] = 0
                        tmp *= 0

                        mlp_g_weight[:,m_out] = mlp_g_weight[:,m_out] * tmp.unsqueeze(0)
                        mlp_u_weight[:,m_out] = mlp_u_weight[:,m_out] * tmp.unsqueeze(0)
                        attn_out_weight[m_out,:] = attn_out_weight[m_out,:] * tmp.unsqueeze(1)

                        cur_layer.mlp.gate_proj.weight.copy_(mlp_g_weight)
                        cur_layer.mlp.up_proj.weight.copy_(mlp_u_weight)
                        cur_layer.self_attn.o_proj.weight.copy_(attn_out_weight)


                    # process attn_V_mask (weight_projection for attn_V_mask)
                    V_mask = torch.cat(m_V)
                    V_mask_repeated = torch.cat([t.repeat(self.num_groups) for t in m_V])
                    ratio = (1 - V_mask).sum() / N_t

                    if ratio > 0:
                        V_mask = (V_mask == 0)
                        V_mask_repeated = (V_mask_repeated == 0)

                        attn_v_weight = cur_layer.self_attn.v_proj.weight.data
                        attn_out_weight = cur_layer.self_attn.o_proj.weight.data

                        if hasattr(self.cfg, "attention_bias") and self.cfg.attention_bias == False:
                            w_norm = attn_v_weight[V_mask, :].pow(2).sum((1))
                            w_norm += attn_out_weight[:, V_mask_repeated].pow(2).sum((0))
                            w_norm = w_norm.add(1e-8).pow(1/2.)

                            attn_v_weight[V_mask,:] = attn_v_weight[V_mask,:] / w_norm.unsqueeze(1)
                            attn_out_weight[:, V_mask_repeated] = attn_out_weight[:, V_mask_repeated] / w_norm.unsqueeze(0)

                            tmp = -self.lam * lr + w_norm#* ratio + w_norm
                            tmp[tmp<0] = 0
                            tmp *=0

                            attn_v_weight[V_mask, :] = attn_v_weight[V_mask,:] * tmp.unsqueeze(1)
                            attn_out_weight[:, V_mask_repeated] = attn_out_weight[:, V_mask_repeated] * tmp.unsqueeze(0)
                            
                            cur_layer.self_attn.v_proj.weight.copy_(attn_v_weight)
                            cur_layer.self_attn.o_proj.weight.copy_(attn_out_weight)

                        # not supported currently
                        else:
                            attn_v_bias   = cur_layer.self_attn.v_proj.bias
                            gl_loss       = ((1 - V_mask).unsqueeze(1) * attn_v_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                            + ((1 - V_mask) * attn_v_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                                            + ((1 - V_mask_repeated).unsqueeze(0) * attn_out_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
                        

                    # process attn_K_mask (Q_mask)
                    K_mask = torch.cat(m_K)
                    Q_mask = torch.cat([t.repeat(self.num_groups) for t in m_K])
                    ratio = (1 - K_mask).sum() / N_t

                    if ratio > 0:
                        attn_k_weight = cur_layer.self_attn.k_proj.weight.data
                        attn_q_weight = cur_layer.self_attn.q_proj.weight.data

                        if hasattr(self.cfg, "attention_bias") and self.cfg.attention_bias == False:
                            m_K_out = (K_mask == 0)
                            m_Q_out = (Q_mask == 0)

                            # Normalize attn_k_weight and attn_q_weight
                            w_norm = attn_k_weight[m_K_out, :].pow(2).sum(dim=1)
                            w_norm += attn_q_weight[m_Q_out, :].pow(2).sum(dim=1)
                            w_norm = w_norm.add(1e-8).pow(0.5)

                            attn_k_weight[m_K_out, :] = attn_k_weight[m_K_out, :] / w_norm.unsqueeze(1)
                            attn_q_weight[m_Q_out, :] = attn_q_weight[m_Q_out, :] / w_norm.unsqueeze(1)

                            # Apply scaling factor
                            tmp = -self.lam * lr + w_norm #* ratio + w_norm
                            tmp = tmp.clamp(min=0)  # Equivalent to tmp[tmp < 0] = 0
                            tmp *= 0

                            attn_k_weight[m_K_out, :] = attn_k_weight[m_K_out, :] * tmp.unsqueeze(1)
                            attn_q_weight[m_Q_out, :] = attn_q_weight[m_Q_out, :] * tmp.unsqueeze(1)

                            # Update weights
                            cur_layer.self_attn.k_proj.weight.copy_(attn_k_weight)
                            cur_layer.self_attn.q_proj.weight.copy_(attn_q_weight)

                        # not supported currently
                        else:
                            attn_k_bias   = cur_layer.self_attn.k_proj.bias
                            attn_q_bias   = cur_layer.self_attn.q_proj.bias
                            gl_loss       = ((1 - K_mask).unsqueeze(1) * attn_k_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                            + ((1 - K_mask) * attn_k_bias).pow(2).add(1e-8).pow(1/2.).sum() \
                                            + ((1 - Q_mask).unsqueeze(1) * attn_q_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                            + ((1 - Q_mask) * attn_q_bias).pow(2).add(1e-8).pow(1/2.).sum()

        return True
    


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
            
            print(gl_loss)

            gl_list.append(torch.tensor(gl_loss.item()).cuda())
            del gl_loss

            # process attn_out_mask
            attn_out_weight = cur_layer.self_attn.o_proj.weight
            gl_loss       = ((1 - m_out).unsqueeze(1) * attn_out_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                            + ((1 - m_out).unsqueeze(0) * mlp_u_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()    \
                            + ((1 - m_out).unsqueeze(0) * mlp_g_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
            
            print(gl_loss)

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
            
            print(gl_loss)

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
                
            print(gl_loss)
            gl_list.append(torch.tensor(gl_loss.item()).cuda())
            del gl_loss
        
        # sum gl_loss (for value tracing only)
        #sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)
        sum_loss = sum(gl_list) / len(gl_list)
        return sum_loss    

        







