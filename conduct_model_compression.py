import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from custom_llms.llama_disp import LlamaForCausalLM  # Import custom LLaMA class
import os
from util_llm import customized_lora_substitution, LoRALinear

def print_banner():
    """ Display a cool ASCII banner when the script starts """
    print("""
                  ██████████
              ████          ████
          ████       stage2:       ████
    ████   acquire the real pruned model   ████
       ████                            ████
          ████                    ████
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

def compress_loralinear(layer, in_mask=None, out_mask=None):
    """
    Compress a LoRALinear layer by applying structured pruning 
    to the original weights and the LoRA parameters.

    Arguments:
    - layer: LoRALinear instance (original model)
    - in_mask: Pruning mask for the input dimension (optional)
    - out_mask: Pruning mask for the output dimension (optional)

    Returns:
    - A new, compressed LoRALinear layer.
    """
    assert isinstance(layer, LoRALinear), "Expected a LoRALinear layer."

    # Determine which dimensions need pruning
    in_dim, out_dim = layer.linear.in_features, layer.linear.out_features
    pruned_in_dim = int(in_mask.sum().item()) if in_mask is not None else in_dim
    pruned_out_dim = int(out_mask.sum().item()) if out_mask is not None else out_dim

    # Compute selected indices for input and output
    select_in_idx = (in_mask == 1).nonzero() if in_mask is not None else torch.arange(in_dim)
    select_out_idx = (out_mask == 1).nonzero() if out_mask is not None else torch.arange(out_dim)
    print(pruned_in_dim, pruned_out_dim)
    # **Compress original weight (linear.weight)**
    pruned_linear = nn.Linear(pruned_in_dim, pruned_out_dim, bias=False)
    pruned_weight = layer.linear.weight.data[select_out_idx, :][:, select_in_idx].squeeze(2)
    pruned_linear.weight.data.copy_(pruned_weight)

    # **Compress LoRA matrices (A and B)**
    pruned_lora_A = nn.Parameter(layer.lora_A[select_in_idx, :])  # Adjust input dimension for LoRA
    pruned_lora_B = nn.Parameter(layer.lora_B[:, select_out_idx])  # Adjust output dimension for LoRA

    # **Create a new compressed LoRALinear layer**
    compressed_layer = LoRALinear(pruned_linear, r=layer.lora_A.shape[1], dropout=layer.lora_dropout.p, svd_init=False)
    compressed_layer.lora_A = pruned_lora_A
    compressed_layer.lora_B = pruned_lora_B

    return compressed_layer


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
    api_token = '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'
    model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
    print(f"\n[INFO] Pretraining TP: {model_cfg.pretraining_tp}")
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'left'
    
    semi_pruned_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        token=api_token
    ).to('cuda')
    semi_pruned_model.resize_token_embeddings(len(tokenizer))

    # Load pruning mask from checkpoint
    # ** we do not merge LoRA for the consideration of quantized pre-trained weights to keep consistency
    print("\n[INFO]: Loading semi-pruned checkpoint...")
    checkpoint = torch.load(atp_disp_ckpt, map_location="cpu")
    customized_lora_substitution(semi_pruned_model, rank=32, dropout=0.1)
    semi_pruned_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    semi_pruned_model.eval()

    # Organize pruning decisions
    print("\n[INFO]: Organizing pruning decisions...")
    cur_mask_vec = checkpoint["mask_vec"].to("cuda")
    masks = transform_output_layer_DISP(cur_mask_vec, model_name)

    # Begin pruning process
    print("\n[INFO]: Replacing LoRALinear layers with compressed versions...")
    for layer_idx, decoder_layer in enumerate(semi_pruned_model.model.layers):
        cur_layer = decoder_layer
        layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in masks]

        # Extract pruning masks for different projections
        m_s1, m_s2, m_s3, m_s4, m_s5 = layer_wise_masks

        # **Compress QKV Projections**
        cur_layer.self_attn.q_proj = compress_loralinear(cur_layer.self_attn.q_proj, in_mask=m_s1)
        cur_layer.self_attn.k_proj = compress_loralinear(cur_layer.self_attn.k_proj, in_mask=m_s1)
        cur_layer.self_attn.v_proj = compress_loralinear(cur_layer.self_attn.v_proj, in_mask=m_s1)

        # **Compress O Projection**
        cur_layer.self_attn.o_proj = compress_loralinear(cur_layer.self_attn.o_proj, out_mask=m_s2)

        # **Compress MLP Layers**
        cur_layer.mlp.gate_proj    = compress_loralinear(cur_layer.mlp.gate_proj, in_mask=m_s3, out_mask=m_s4)
        cur_layer.mlp.up_proj      = compress_loralinear(cur_layer.mlp.up_proj, in_mask=m_s3, out_mask=m_s4)
        cur_layer.mlp.down_proj    = compress_loralinear(cur_layer.mlp.down_proj, in_mask=m_s4, out_mask=m_s5)

    print("\n[INFO]: Pruning and compression complete! Saving compressed model...")
    print(semi_pruned_model)

    # Save compressed model
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    torch.save({"model_state_dict": semi_pruned_model.state_dict()}, output_ckpt)

    print(f"\n[INFO]: Compressed LoRA model saved at: {output_ckpt}")


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