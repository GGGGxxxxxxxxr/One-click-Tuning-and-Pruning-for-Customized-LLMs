import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_llms.llama_disp import LlamaForCausalLM  # Import custom LLaMA class
import os


def print_banner():
    """ Display a cool ASCII banner when the script starts """
    print("""
                  ██████████
              ████          ████
          ████       ATP:       ████
       ████   All-in-One Tuning     ████
       ████    & Structural Pruning ████
          ████    version 2.0    ████
              ████          ████
                  ██████████
    """)


def transform_output_layer_DISP(inputs, model_name):
    """ 
    Process the binary mask vector based on model architecture. 
    This function ensures that the correct pruning decisions are applied 
    to different projection layers of LLaMA.
    """
    if model_name == 'llama2-7b':
        lw_structure = [128, 4096, 4096, 11008, 4096]
        num_kv_heads = 32
    elif model_name == 'llama3-8b':
        lw_structure = [128, 128, 14336]
        num_kv_heads = 8
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    arch_vector = []
    start = 0
    for i, size in enumerate(lw_structure):
        end = start + size
        sliced_input_tensor = inputs[:, start:end]
        if i < 1:  # Extend KV head mask across all heads
            replicated_slices = sliced_input_tensor.repeat(1, num_kv_heads)
            arch_vector.append(replicated_slices)
        else:
            arch_vector.append(sliced_input_tensor)
        start = end
    return arch_vector


def prune_llama(
    model_name,
    atp_disp_ckpt,
    output_ckpt
):
    """ 
    Load the semi-pruned LLaMA model, apply structural pruning, and save the compressed model. 
    """
    print_banner()
    print(f"\n[INFO]: Loading model `{model_name}`...")

    if model_name == 'llama2-7b':
        api_token = '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'
        hf_model_name = "meta-llama/Llama-2-7b-hf"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'left'

    # Load LLaMA model
    semi_pruned_model = LlamaForCausalLM.from_pretrained(
        hf_model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        token=api_token,
    ).to('cuda')

    # Load pruning mask from checkpoint
    print("\n[INFO]: Loading semi-pruned checkpoint...")
    checkpoint = torch.load(atp_disp_ckpt, map_location="cpu")
    semi_pruned_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    semi_pruned_model.eval()

    # Organize pruning decisions
    print("\n[INFO]: Organizing pruning decisions...")
    cur_mask_vec = checkpoint["mask_vec"].to("cuda")
    masks = transform_output_layer_DISP(cur_mask_vec, model_name)

    # Begin pruning process
    print("\n[INFO]: Replacing projections with pruned versions...")
    for layer_idx, decoder_layer in enumerate(semi_pruned_model.model.layers):
        cur_layer = decoder_layer
        layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in masks]

        # Extract pruning masks for different projections
        m_s1, m_s2, m_s3, m_s4, m_s5 = layer_wise_masks

        # **Prune QKV Projections**
        attn_in_dim = int(m_s1.sum().item())  # New input dimension after pruning
        select_index = (m_s1 == 1).nonzero().squeeze()
        pruned_q_proj = nn.Linear(attn_in_dim, 4096, bias=False)
        pruned_k_proj = nn.Linear(attn_in_dim, 4096, bias=False)
        pruned_v_proj = nn.Linear(attn_in_dim, 4096, bias=False)

        # Copy the pruned weights
        pruned_q_proj.weight.data.copy_(cur_layer.self_attn.q_proj.weight.data[:, select_index])
        pruned_k_proj.weight.data.copy_(cur_layer.self_attn.k_proj.weight.data[:, select_index])
        pruned_v_proj.weight.data.copy_(cur_layer.self_attn.v_proj.weight.data[:, select_index])

        # Replace the original layers with pruned versions
        cur_layer.self_attn.q_proj = pruned_q_proj
        cur_layer.self_attn.k_proj = pruned_k_proj
        cur_layer.self_attn.v_proj = pruned_v_proj

        # **Prune O Projection**
        o_out_dim = int(m_s2.sum().item())  # New output dimension
        select_index = (m_s2 == 1).nonzero().squeeze()
        pruned_o_proj = nn.Linear(4096, o_out_dim, bias=False)
        pruned_o_proj.weight.data.copy_(cur_layer.self_attn.o_proj.weight.data[select_index, :])
        cur_layer.self_attn.o_proj = pruned_o_proj

        # **Prune MLP Layers**
        mlp_in_dim = int(m_s3.sum().item())
        mlp_inter_dim = int(m_s4.sum().item())
        mlp_out_dim = int(m_s5.sum().item())

        select_in_index = (m_s3 == 1).nonzero().squeeze()
        select_inter_index = (m_s4 == 1).nonzero().squeeze()
        select_out_index = (m_s5 == 1).nonzero().squeeze()

        pruned_mlp_u_proj = nn.Linear(mlp_in_dim, mlp_inter_dim, bias=False)
        pruned_mlp_g_proj = nn.Linear(mlp_in_dim, mlp_inter_dim, bias=False)
        pruned_mlp_d_proj = nn.Linear(mlp_inter_dim, mlp_out_dim, bias=False)

        pruned_mlp_g_proj.weight.data.copy_(cur_layer.mlp.gate_proj.weight.data[select_inter_index, select_in_index])
        pruned_mlp_u_proj.weight.data.copy_(cur_layer.mlp.up_proj.weight.data[select_inter_index, select_in_index])
        pruned_mlp_d_proj.weight.data.copy_(cur_layer.mlp.down_proj.weight.data[select_out_index, select_inter_index])

        cur_layer.mlp.gate_proj = pruned_mlp_g_proj
        cur_layer.mlp.up_proj = pruned_mlp_u_proj
        cur_layer.mlp.down_proj = pruned_mlp_d_proj

    print("\n[INFO]: Pruning complete! Saving pruned model...")

    # Save pruned model
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    torch.save({"model_state_dict": semi_pruned_model.state_dict()}, output_ckpt)

    print(f"\n[INFO]: Pruned model saved at: {output_ckpt}")


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Prune LLaMA model and save compressed checkpoint.")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Model name (e.g., llama2-7b)")
    parser.add_argument("--atp_disp_ckpt", type=str, required=True, help="Path to semi-pruned model checkpoint.")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Path to save the fully pruned model checkpoint.")

    args = parser.parse_args()

    # Run pruning function
    prune_llama(
        model_name=args.model_name,
        atp_disp_ckpt=args.atp_disp_ckpt,
        output_ckpt=args.output_ckpt
    )