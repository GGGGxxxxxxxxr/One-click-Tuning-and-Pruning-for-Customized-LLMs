import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from typing import List
from transformers import AutoConfig
import torch.nn.init as init
import math
#-----------------------------------------------------------------#
# counting prunable structures for Decoder-only-based transformer LLMs
# We do not prune the nn.Embedding following other works!
# head-wise counting rules:
# each Q-split is recognized as a prunable head! 
# such design is to align MHA and its variant GroupAttention in structure pruning.
# >>>>> ---------------------------------------------------- <<<<<#
# fine-grained counting rules:
# Q, K mask is equivalent thus only WK mask is considered.
# We prefer K_mask due to the purpose of Multi-head Attn & Grouped-Query Attn Alignment. Q_mask would lead to a un-prunable K_weight, thus would lead to potentional mismatch between: 
# **[masked LLM via generated structural mask] & [GroupLasso-regularized true LLM]
# Besides, WV, WOut, MLPUp is considered as prunable structure.
# RMSNorm has a trainable weight (1, hiddenDim), but its mask is equivalent to the WFFDown, thus we dont count it.
# **MLP_DownProjection is not considered as a prunable structure so that we could make sure the output dimension of each DecoderLayer is still [hidden_dim], only the inner block dimension would be changed.
# >>>>> ---------------------------------------------------- <<<<<#
# We follow the ATO's design to configure the output Prunable_Structure as a List[] of int, indicating the prunbale dimension for each structure.
# E.g:
# for 1-layer traditional MHA with Head = 2, head-dim = 64, MLPMultiplier = 4
# p_structures = [64, 64, 64, 64, 64, 64*4] (indicating WK_1, WK_2, WV_1, WV_2, WOut, MLPUp) (with fine-grained pruning)
# p_structures = [2, 64, 64*4]              (indicating Q-head-wise-prune, WOut, MLPUp)      (with head-wise pruning)
# ****************************
# ** modified function (10/22): 
# ** add new feature of pruning_scheme selection as 'LAYER_UNIFORM_ATTN'!
# ** a unified layer-specific K_head (or V_head) mask would be generated and all attn_heads within this layer would share this exact same mask to ensure the attention uniform shape 
def count_llm_p_structures(model: nn.Module, model_config: AutoConfig, pruning_scheme: str) -> List[int]:
    num_layers   = model_config.num_hidden_layers       # num_of_layers
    num_heads    = model_config.num_attention_heads     # num_of_Q_Splits
    num_kv_heads = model_config.num_key_value_heads     # num_of_KV_Splits 
    hidden_size  = model_config.hidden_size             # hidden_dim
    head_dim     = hidden_size // num_heads             # head_dim
    intermediate_size = model_config.intermediate_size  # MLP up_projection_dim

    # Print ModelConfig
    print("=====> Counting Prunable Structure based on the ModelConfig: <=====")
    print(f"Number of Layers: {num_layers}")
    print(f"Number of Q Splits (Heads): {num_heads}")
    print(f"Number of KV Splits: {num_kv_heads}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Head Dimension: {head_dim}")
    print(f"MLP UpProjection Dimension: {intermediate_size}")

    # 1. append TOTAL_LAYERS as the first element
    p_structure = []
    p_structure.append(num_layers)
    
    # layer-wise prunable structure holder
    lw_structure = []

    # 2. counting prunable structures based on pruning methods within in a single layer
    if pruning_scheme == 'inner':
        print("=====> Counting Fine-grained Prunable Structures. <=====")
        # a) [K_split_1, ..., K_split_n] 
        #    [V_split_1, ..., V_split_n] (n = self.config.num_key_value_heads)
        for _ in range(2 * num_kv_heads):
            lw_structure.append(head_dim)
        # b) Attn_Output
        lw_structure.append(hidden_size)
        # c) MLP_Up
        lw_structure.append(intermediate_size)
        
        print("=====> Prunable Structure for One-Layer: <=====")
        print(lw_structure)
    elif pruning_scheme == 'layer_uniform_attn':
        print("=====> Counting Layer_Uniform_Attn Prunable Structures. <=====")
        # a) [Unified K_split] [Unified V_split]
        lw_structure.append(head_dim)
        lw_structure.append(head_dim)

        # b) attn_output
        # ** we have revisited the implentation details, as for the residual-link connection capability after pruning,
        # ** dimensional pruning on the out_projection would lead to mismatch between the dimension of  [residual] & [hidden_state]!
        #lw_structure.append(hidden_size)

        # c) MLP_Up
        lw_structure.append(intermediate_size)

        print("=====> Prunable Structure for One-Layer: <=====")
        print(lw_structure)
    else:
        print("=====> Not implemented yet!. <=====")

    # 3. combine as pruning indicator
    p_structure.append(lw_structure)

    # 4. [Optional] if 'layer_uniform_attn', we attach the {num_of_kv_heads} at the tail for future access
    if pruning_scheme == 'layer_uniform_attn':
        p_structure.append(num_kv_heads)

    print("=====> LLM p_structure counting finished. <=====")
    print("Prunable Structure:", p_structure)
    
    return p_structure
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# List[] of attn_mask into applicable Q, V masks
def attn_mask_transformation(mask_list, num_heads, bsz, q_len, head_dim):
    """
    Perform transformation on mask_list, converting it into a tensor of shape [bsz, num_heads, q_len, head_dim].
    
    Parameters:
    - mask_list: A list containing num_heads tensors, each with shape [head_dim]
    - num_heads: The number of heads (e.g., self.num_heads or self.num_key_value_heads)
    - bsz: Batch size
    - q_len: Sequence length
    - head_dim: The dimensionality of each head
    """
    # Stack the list into a tensor of shape [num_heads, head_dim]
    masks = torch.stack(mask_list)

    # Reshape to [num_heads, 1, head_dim], then expand to [num_heads, q_len, head_dim]
    masks = masks.view(num_heads, 1, head_dim).expand(num_heads, q_len, head_dim)

    # Expand to [bsz, num_heads, q_len, head_dim]
    masks = masks.unsqueeze(0).expand(bsz, num_heads, q_len, head_dim)

    return masks
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
## ** DEPRECIATED
# version.0.2 update: We no longer use pruning_contribution for such sparsity control
# ** remember: each dimensional mask [0/1] would possibly result in the disabling of the current weight matrix & the sequantial weight matrix!
# ** so that a ratio ought to take care both **
# calculate pruning ratio contribution for each individual [0/1] (masked-out dimension indicator) based on model.Config
def pruning_ratio_contribution(model_cfg):
    num_key_value_heads = model_cfg.num_key_value_heads
    num_attention_heads = model_cfg.num_attention_heads
    num_kv_groups = num_attention_heads / num_key_value_heads
    multiplier = model_cfg.intermediate_size / model_cfg.hidden_size

    k_mask_ratio = 1 + num_kv_groups
    v_mask_ratio = 1 + num_attention_heads
    o_mask_ratio = num_attention_heads * (1 + multiplier)
    u_mask_ratio = num_attention_heads * (1 + 2 * multiplier)

    return {
        "k_ratio": k_mask_ratio,
        "v_ratio": v_mask_ratio,
        "o_ratio": o_mask_ratio,
        "u_ratio": u_mask_ratio
    }

#-----------------------------------------------------------------#

## ** UPDATED
# version.0.2: we now use more accurate param counting for sparsity control, as number of params is our main pruning focus.
# this function is for the total params counting 
def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    def __init__(self, linear_module, r=8, dropout=0.1):
        super(LoRALinear, self).__init__()
        # 保留原始的 Linear 层
        self.linear = linear_module
        in_features = linear_module.in_features
        out_features = linear_module.out_features
        data_type = linear_module.weight.dtype  # 确保 dtype 一致
        cur_device = linear_module.weight.device  # 确保 device 一致
        
        # 初始化 LoRA 参数
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=cur_device))
        init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # 使用 Kaiming Uniform 初始化
        
        # lora_B 零初始化
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=cur_device))
        
        # dropout for LoRA
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 原始 Linear 层输出 + LoRA 路径
        if mask == None:
            original_output = self.linear(x)
            lora_output = self.lora_dropout(x @ self.lora_A.T) @ self.lora_B.T
            return original_output + lora_output
        else:
            original_output = self.linear(x) * mask
            lora_output = self.lora_dropout(x @ self.lora_A.T) @ self.lora_B.T
            return original_output + lora_output

# 递归替换 decoder_layer 中的所有 Linear 层为 LoRALinear
def replace_linear_with_lora(decoder_layer, rank=8, dropout=0.1):
    replacement_count = 0  # 记录每层替换的 Linear 数量
    
    # 使用 named_modules() 递归查找所有子模块
    for name, module in decoder_layer.named_modules():
        if isinstance(module, nn.Linear):
            # 替换为自定义的 LoRALinear
            lora_linear = LoRALinear(module, r=rank, dropout=dropout)
            
            # 获取父模块，并将该 Linear 层替换为 LoRALinear
            parent_module = dict(decoder_layer.named_modules())[name.rsplit('.', 1)[0]]
            setattr(parent_module, name.split('.')[-1], lora_linear)
            
            replacement_count += 1

    return replacement_count

# 主函数：冻结整个模型，并将 Linear 替换为 LoRALinear
def customized_lora_substitution(llm_model, rank=8, dropout=0.1):
    # 1. 冻结整个模型的参数
    for param in llm_model.parameters():
        param.requires_grad = False

    # 2. 遍历每个 decoder 层，替换 Linear 为 LoRALinear
    for layer_idx, decoder_layer in enumerate(llm_model.model.layers):
        # 替换当前层的 Linear 模块为 LoRALinear，并获取该层的替换数量
        replacement_count = replace_linear_with_lora(decoder_layer, rank=rank, dropout=dropout)
        
        # 每个 decoder layer 中应有 7 个 Linear，进行断言
        assert replacement_count == 7, f"Expected 7 Linear layers, but replaced {replacement_count} in layer {layer_idx}"

    '''
    trainable_params = [(name, param) for name, param in llm_model.named_parameters() if param.requires_grad]
    print("Trainable parameters after LoRA Infusion:")
    for name, param in trainable_params:
        print(f"{name}: {param.shape}")
    '''
    
    print("All Linear layers in decoder have been replaced with LoRALinear.")





    

        



