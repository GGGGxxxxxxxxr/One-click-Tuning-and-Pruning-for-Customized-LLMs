from __future__ import absolute_import
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    #U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, T, offset=0):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()

    y = logits + gumbel_sample + offset
    return torch.sigmoid(y / T)


def hard_concrete(out):
    device = out.device
    out_hard = torch.zeros(out.size())
    out_hard[out>=0.5]=1
    if out.is_cuda:
        out_hard = out_hard.to(device)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    out_hard = (out_hard - out).detach() + out
    return out_hard

def truncate_normal_(size, a=-1, b=1):
    values = truncnorm.rvs(a,b,size=size)
    values = torch.from_numpy(values).float()
    return values

class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #train = train
        ctx.grad_w = grad_w

        input_clone = input.clone()
        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()

        gw = ctx.grad_w
        # print(gw)
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda()

        return grad_input * gw, None, None

#-----------------------------------------------------------------#
# Mask Controller Network Design
# >>>>> ---------------------------------------------------- <<<<<#
# version 1.0:
# modified based on ATO's HyperStructure
# we forked input [len(p_structure), 1, 64] into [len(p_structure), LAYERS, 64] where bsz is increased from 1 to LAYERS, thus we could have parallel processing for layer-wise pruning mask.
# so that each layer in LLMs would have it individual input mask params
# the forked input would share a List[] of LinearProjection into Layer-wise weight mask
# >>>>> ---------------------------------------------------- <<<<<#
# **notice: currently we employ seperate linear projections for each prunable linear, 
# but we would further investigate whether it is possible to delopy a uniform projection for each decoderlayer

## ** depreciated: GRU-based hypernet() is discarded right now, please refer to the following encoderblock-based design.
class LLM_HyperStructure_old(nn.Module):
    def __init__(self, p_structure=None, T=0.4, base=3, args=None):
        super(LLM_HyperStructure, self).__init__()
        # >>>>> ---------------------------------------------------- <<<<<#
        self.num_layers   = p_structure[0]
        self.lw_structure = p_structure[1]
        # >>>>> ---------------------------------------------------- <<<<<#

        # >>>>> ---------------------------------------------------- <<<<<#
        # network core
        self.T       = T  
        self.base    = base                                                 # decay
        
        self.ln      = nn.LayerNorm([256])                                  # layernorm

        self.Bi_GRU  = nn.GRU(64, 128, bidirectional=True)                  # [input dim, output_dim]
        self.h0      = torch.zeros(2, self.num_layers, 128).to(dtype=torch.bfloat16)                 # [bidirect, LAYERS, output_dim]

        inputs = torch.empty(len(self.lw_structure), self.num_layers, 64, dtype=torch.float32)
        # Step 2: Apply orthogonal initialization in float32
        nn.init.orthogonal_(inputs)
        # Step 3: Convert to bfloat16 after initialization
        inputs = inputs.to(dtype=torch.bfloat16)
        # Step 4: Detach the tensor to ensure no gradient tracking
        self.inputs = inputs.detach()                              # 'input' itself is a untrainable nn.param
        
        self.linear_list = [nn.Linear(256, self.lw_structure[i], bias=False) 
                            for i in range(len(self.lw_structure))]         # project to linear_outputDim for prunable LinearWeight in each LLM layer
        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        

        # Initialize parameters
        #self.initialize_parameters()
    # >>>>> ---------------------------------------------------- <<<<<#

    # >>>>> ---------------------------------------------------- <<<<<#
    def initialize_parameters(self):
        # Initialize Linear layers
        for linear_layer in self.mh_fc:
            nn.init.constant_(linear_layer.weight, 1.0)
        
        # Initialize GRU
        for name, param in self.Bi_GRU.named_parameters():
            if 'weight' in name:
                nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize LayerNorm
        nn.init.constant_(self.ln.weight, 1.0)
        nn.init.constant_(self.ln.bias, 0.0)
    # >>>>> ---------------------------------------------------- <<<<<#

    # >>>>> ---------------------------------------------------- <<<<<#
    # modified forward() for LLM-layerwise mask
    def forward(self, dummy):
        # device uniform
        if self.ln.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0     = self.h0.cuda()
        # Bi-GRU
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)                     # [sequence_len, LAYERS, output_dim * 2]
        # layer-wise mask projection
        outputs = [F.relu(self.ln(outputs[i,:]))  for i in  range(len(self.lw_structure))]
        outputs = [self.mh_fc[i](outputs[i])      for i in  range(len(self.mh_fc))]         # List[] of [ ... [LAYERS, output_dim *2] ... ]

        # logits formulation
        out = torch.cat(outputs, dim=1)                                     # sum channel size [LAYERS, total_prunable_dim_in_one_LLM_layer] for Gumble_Softmax(sigmoid)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)        # transform into Gumble_sampled logits 

        if not self.training:
            out = hard_concrete(out)                                        # binary_mask generation
        
        return out                                                          # [LAYERS, total_prunable_dim_in_one_LLM_layer] {.squeeze() has been removed cuz we have multi layers serving as individual bsz.}
    # >>>>> ---------------------------------------------------- <<<<<#

    # >>>>> ---------------------------------------------------- <<<<<#
    # transform [LAYERS, total_prunable_dim_in_one_LLM_layer] into List[] of seperate prunable structures
    # e.g, [..[LAYERS, Q_head_1], ... , [LAYERS, V_head_1], ... , [LAYERS, Output], ...]
    def transform_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.lw_structure)):
            end = start + self.lw_structure[i]
            arch_vector.append(inputs[:, start:end])
            start = end

        return arch_vector
    # >>>>> ---------------------------------------------------- <<<<<#
    
    # >>>>> ---------------------------------------------------- <<<<<#
    # transform dim_mask into applicable mask
    # as LinearProjection output is [bsz, linear_out] (very simple)
    # thus we just need to direct * [layer_idx, binary_dim_mask] with LProj output
    # this function is depreciated in ATO's LLM application
    def vector2mask(self, inputs):
        return None
    # >>>>> ---------------------------------------------------- <<<<<#
    

class LLM_HyperStructure(nn.Module):
    def __init__(self, p_structure=None, T=0.4, base=3, args=None, num_layers=2, num_heads=4):
        super(LLM_HyperStructure, self).__init__()

        # >>>>> Configuration Setup <<<<<#
        self.pruning_scheme = args.pruning_method
        self.num_layers     = p_structure[0]  # Number of layers in the LLM
        self.lw_structure   = p_structure[1]  # Structure of each layer's mask

        # notice: 'num_kv_heads' is not the apporiate term for 'DISP', but we just use it for simplity :)
        if self.pruning_scheme in ['layer_uniform_attn', 'DISP']:
            assert len(p_structure) == 3, "mismatch between prunable_structure holder and the HyperNet() requirements"
            self.num_kv_heads = p_structure[-1]

        self.T              = T  # Temperature for Gumbel-Softmax
        self.base           = base  # Offset for Gumbel-Softmax sampling

        # >>>>> Transformer Encoder <<<<<#
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=num_heads, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable Input Embeddings
        inputs = torch.full((self.num_layers, 64), fill_value=1.5, dtype=torch.float32)
        nn.init.orthogonal_(inputs)
        self.inputs = nn.Parameter(inputs.to(dtype=torch.bfloat16), requires_grad=False)

        # Layer Normalization
        self.ln = nn.LayerNorm(64)

        # Layer-wise Mask Projection
        # version2.0: currently disabled, would investigate in the future 
        '''
        self.mh_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.lw_structure[i])
            )
            for i in range(len(self.lw_structure))
        ])
        '''

        self.mh_fc_list = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(64, self.lw_structure[i], bias=False)  # Multiple linear layers for each mask projection
                for i in range(len(self.lw_structure))
            ]) for _ in range(self.num_layers)  # One list per layer
        ])

    def forward(self, dummy=None):
        # >>>>> Device Management <<<<<#
        device = next(self.parameters()).device
        self.inputs = self.inputs.to(device)

        # >>>>> Transformer Encoder <<<<<#
        transformer_out = self.transformer(self.inputs)  # Shape: (num_layers, 64)
        print(transformer_out.dtype)
        # Apply Layer Normalization
        norm_out = self.ln(transformer_out)
        print(norm_out.dtype)
        '''
        # >>>>> Layer-Wise Mask Projection <<<<<#
        outputs = [fc(norm_out) for fc in self.mh_fc]
        out = torch.cat(outputs, dim=-1)  # Shape: (num_layers, total_mask_dim)
        '''

        outputs = []
        for layer_idx in range(self.num_layers):
            layer_out = norm_out[layer_idx].unsqueeze(0)  # Shape: (1, 64)
            layer_masks = [
                linear(layer_out)  # Apply each linear projection for this layer
                for linear in self.mh_fc_list[layer_idx]
            ]
            outputs.append(torch.cat(layer_masks, dim=-1))  # Concatenate masks for this layer

        # Concatenate all layers' outputs
        out = torch.cat(outputs, dim=0)  # Shape: (32, total_mask_dim)
        print(out.dtype)
        # >>>>> Gumbel-Softmax Sampling <<<<<#
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)
        print(out.dtype)
        # Convert to Binary Mask in Evaluation Mode
        if not self.training:
            out = hard_concrete(out)
        print(out.dtype)
        return out

    # convert output vector into applicable masks for LLM maksed inference
    def transform_output(self, inputs):
        """Transform concatenated mask vector into individual layer masks."""
        if self.pruning_scheme == 'inner':
            arch_vector = []
            start = 0
            for size in self.lw_structure:
                end = start + size
                arch_vector.append(inputs[:, start:end])
                start = end

            return arch_vector
        
        elif self.pruning_scheme == 'atp_layer_uniform_attn':
            arch_vector = []
            start = 0
            for i, size in enumerate(self.lw_structure):
                end                 = start + size
                sliced_input_tensor = inputs[:, start : end]

                ## **
                ## we need to extend K_V_head_mask for the whole layer (multi-head)
                ## for K, V mask, we repeat a layer-uniform [head-wise] pruning for the actual K_proj / V_proj mask for more efficient implementation
                if i < 2:  
                    #replicated_slices = [sliced_input_tensor] * self.num_kv_heads
                    #arch_vector.extend(replicated_slices)
                    replicated_slices = sliced_input_tensor.repeat(1, self.num_kv_heads)
                    arch_vector.append(replicated_slices)
                else:
                    arch_vector.append(sliced_input_tensor)
                start = end
            assert len(arch_vector) == 3, "K(Q), V , MLP_up(Gate) masks are expected, 3 seperate masks in total, please check."
            return arch_vector
        
        # version2.0: DISP mask conversion
        elif self.pruning_scheme == 'DISP':
            arch_vector = []
            start = 0
            for i, size in enumerate(self.lw_structure):
                end                 = start + size
                sliced_input_tensor = inputs[:, start : end]

                ## ** in DISP pruning space, S1 is applicable for EACH head-attention input dimension(Q,K,V), thus we simply repeat head-wise S1 into S1 x num_heads
                ## notice: this logic is different from 'layer_uniform_attn'!
                if i == 0:  
                    replicated_slices = sliced_input_tensor.repeat(1, self.num_kv_heads)
                    arch_vector.append(replicated_slices)
                else:
                    arch_vector.append(sliced_input_tensor)
                start = end
            assert len(arch_vector) == 5, "S1(extened), S2, S3, S4, S5 masks are expected, 5 seperate masks in total, please check."
            return arch_vector
                
    
    # depricated right now, refer to func: 'transform_output'
    def vector2mask(self, inputs):
        """(Deprecated) Transform vector to mask - not used in current ATO's application."""
        return None

if __name__ == '__main__':
    net = LLM_HyperStructure()
    y = net()
    print(y)
