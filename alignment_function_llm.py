import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from math import sqrt, floor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
        self.grad_mul    = args.grad_mul
        self.lam         = args.gl_lam
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

        # adjust regularization tensity
        if epoch >= 1:
            self.lam = 1000 * self.lam

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

                '''
                # test only
                random_tensor = torch.randint(0, 2, m_umlp.size()).cuda()
                m_umlp = random_tensor
                '''

                # process MLP_up_mask
                mlp_g_weight = cur_layer.mlp.gate_proj.weight
                mlp_u_weight = cur_layer.mlp.up_proj.weight
                mlp_d_weight = cur_layer.mlp.down_proj.weight
                gl_loss      = ((1 - m_umlp).unsqueeze(1) * mlp_g_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_umlp).unsqueeze(1) * mlp_u_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_umlp).unsqueeze(0) * mlp_d_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
                
                gl_list.append(torch.tensor(gl_loss.item()).cuda())
                self.scaler.scale(gl_loss * self.lam).backward()


                # process attn_out_mask
                attn_out_weight = cur_layer.self_attn.o_proj.weight
                gl_loss       = ((1 - m_out).unsqueeze(1) * attn_out_weight).pow(2).sum((1)).add(1e-8).pow(1/2.).sum() \
                                + ((1 - m_out).unsqueeze(0) * mlp_u_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()    \
                                + ((1 - m_out).unsqueeze(0) * mlp_g_weight).pow(2).sum((0)).add(1e-8).pow(1/2.).sum()
                
                gl_list.append(torch.tensor(gl_loss.item()).cuda())
                self.scaler.scale(gl_loss * self.lam).backward()

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
                self.scaler.scale(gl_loss * self.lam).backward()

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
                self.scaler.scale(gl_loss * self.lam).backward()

                '''
                device = torch.cuda.current_device()
                print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
                '''
                
        # sum gl_loss (for value tracing only)
        #sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)
        sum_loss = sum(gl_list) / len(gl_list)
        return sum_loss              
                

              







