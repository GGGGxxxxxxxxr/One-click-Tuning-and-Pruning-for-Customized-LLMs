from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from custom_llms.llama_disp import LlamaForCausalLM
import sys
import torch

import time
import tqdm
import math


def transform_output_layer_DISP(inputs, model_name=None):
    if model_name  == 'llama2-7b':
        lw_structure = [128] + [4096] + [4096] + [11008] + [4096]
        num_kv_heads = 32
    elif model_name == 'llama3-8b':
        lw_structure = [128] * 2 + [14336]
        num_kv_heads = 8

    arch_vector = []
    start = 0
    for i, size in enumerate(lw_structure):
        end = start + size
        sliced_input_tensor = inputs[:, start:end]

        if i < 1:  # Extend K_V_head_mask for the whole layer (multi-head)
            replicated_slices = sliced_input_tensor.repeat(1, num_kv_heads)
            arch_vector.append(replicated_slices)
        else:
            arch_vector.append(sliced_input_tensor)
        start = end
    return arch_vector


def main(
    atp_disp_ckpt: str = '/orange/sgao1/atp_llm_dir/llama2_7b_0.5_alpacagpt_basic.pth.tar',
    out_dir: str = '/orange/sgao1/',
    model_name: str = 'llama',
):
    print("\n[INFO]: Your semi-pruned-model would be compressed into the real-pruned model now.")

    if model_name == 'llama2-7b':

        print("\n[INFO]: llama2-7b detected...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'left'
        
        api_token = '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'
        semi_pruned_model = LlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                token=api_token
            ).to('cuda')
        
        print("\n[INFO]: Loading semi-pruned checkpoint...")
        semi_pruned_checkpoint = torch.load(atp_disp_ckpt, map_location=torch.device('cpu'))
        semi_pruned_model.load_state_dict(semi_pruned_model["model_state_dict"], strict=True)
        semi_pruned_model.eval()

        print("\n[INFO]: Merge Semi-pruned LoRA Weights ...")
        

        print("\n[INFO]: Organizing pruning decisions...")
        cur_mask_vec = semi_pruned_checkpoint["mask_vec"].to("cuda")
        masks = transform_output_layer_DISP(cur_mask_vec, model_name=model_name)

        print("\n[INFO]: replacing Projections into Pruned Projections...")
        print("\n")
        for layer_idx, decoder_layer in enumerate(semi_pruned_model.model.layers):
            cur_layer = decoder_layer
            layer_wise_masks = [individual_mask[layer_idx,:] for individual_mask in masks]

            # gather m_1 -- m_5
            m_s1 = layer_wise_masks[0]
            m_s2 = layer_wise_masks[1]
            m_s3 = layer_wise_masks[2]
            m_s4 = layer_wise_masks[3]
            m_s5 = layer_wise_masks[4]

            # process QKV_projections (m_s1)
            # create placeholder for the pruned QKV linear
            attn_in_dim  = int(m_s1.sum().item())
            select_index = (m_s1 == 1).nonzero().squeeze() 
            pruned_q_proj = torch.nn.Linear(attn_in_dim, 4096, bias=False)
            pruned_k_proj = torch.nn.Linear(attn_in_dim, 4096, bias=False)
            pruned_v_proj = torch.nn.Linear(attn_in_dim, 4096, bias=False)
            # gather pruned weights
            pruned_q_weight = cur_layer.self_attn.q_proj.weight.data[:, select_index]
            pruned_k_weight = cur_layer.self_attn.k_proj.weight.data[:, select_index]
            pruned_v_weight = cur_layer.self_attn.v_proj.weight.data[:, select_index]   
            # copy weights
            pruned_q_proj.weight.data.copy_(pruned_q_weight)
            pruned_k_proj.weight.data.copy_(pruned_k_weight)
            pruned_v_proj.weight.data.copy_(pruned_v_weight)
            # replace semi-pruned linear with pruned linear
            cur_layer.self_attn.q_proj = pruned_q_proj
            cur_layer.self_attn.k_proj = pruned_k_proj
            cur_layer.self_attn.v_proj = pruned_v_proj

            # process O_projection (m_s2)
            o_out_dim = int(m_s2.sum().item())
            select_index = (m_s2 == 1).nonzero().squeeze()
            pruned_o_proj = torch.nn.Linear(4096, o_out_dim, bias=False)
            # gather pruned weights
            pruned_o_weight = cur_layer.self_attn.o_proj.weight.data[select_index, :]
            # copy weights
            pruned_o_proj.weight.data.copy_(pruned_o_weight)
            # replace semi-pruned linear with pruned linear
            cur_layer.self_attn.o_proj = pruned_o_proj

            # process MLP (m_s3, m_s4, m_s5)
            mlp_in_dim           = int(m_s3.sum().item())
            mlp_intermediate_dim = int(m_s4.sum().item())
            mlp_out_dim          = int(m_s5.sum().item())
            select_in_index      = (m_s3 == 1).nonzero().squeeze()
            select_inter_index   = (m_s4 == 1).nonzero().squeeze()
            select_out_index     = (m_s5 == 1).nonzero().squeeze()
            pruned_mlp_u_proj    = torch.nn.Linear(mlp_in_dim, mlp_intermediate_dim, bias=False)
            pruned_mlp_g_proj    = torch.nn.Linear(mlp_in_dim, mlp_intermediate_dim, bias=False)
            pruned_mlp_d_proj    = torch.nn.Linear(mlp_intermediate_dim, mlp_out_dim, bias=False)

            # gather pruned weights
            pruned_gate_weight   = cur_layer.mlp.gate_proj.weight.data[select_inter_index, select_in_index]
            pruned_up_weight     = cur_layer.mlp.up_proj.weight.data[select_inter_index, select_in_index]
            pruned_down_weight   = cur_layer.mlp.down_proj.weight.data[select_out_index, select_inter_index]
            # copy weights
            pruned_mlp_u_proj.weight.data.copy_(pruned_up_weight)
            pruned_mlp_g_proj.weight.data.copy_(pruned_gate_weight)
            pruned_mlp_d_proj.weight.data.copy_(pruned_down_weight)
            # replace semi-pruned linear with pruned linear
            cur_layer.mlp.gate_proj = pruned_mlp_g_proj
            cur_layer.mlp.up_proj   = pruned_mlp_u_proj
            cur_layer.mlp.down_proj = pruned_mlp_d_proj

            # print pruned model for visualization
            print(semi_pruned_model)
            
            # save_ckpt
            semi_pruned_model.cpu()
            state_dict = semi_pruned_model.state_dict()


            


            





        

    