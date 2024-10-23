import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from custom_llms.qwen2 import Qwen2ForCausalLM  # Ensure this import is correct
from custom_llms.llama import LlamaForCausalLM
from alignment_function_llm import Group_Lasso_regularization
from sklearn.metrics import precision_recall_fscore_support

def transform_output(inputs):
    lw_structure = [128] * 64 + [4096] + [11008]
    arch_vector = []
    start = 0
    for i in range(len(lw_structure)):
        end = start + lw_structure[i]
        arch_vector.append(inputs[:, start:end])
        start = end
    return arch_vector

def transform_output_layer_uniform(inputs):
    lw_structure = [128] * 2 + [4096] + [11008]
    num_kv_heads = 32
    arch_vector = []
    start = 0
    for i, size in enumerate(lw_structure):
        end = start + size
        sliced_input_tensor = inputs[:, start:end]

        if i < 2:  # Extend K_V_head_mask for the whole layer (multi-head)
            replicated_slices = [sliced_input_tensor] * num_kv_heads
            arch_vector.extend(replicated_slices)
        else:
            arch_vector.append(sliced_input_tensor)
        start = end
    return arch_vector

def initialize_model_and_tokenizer():
    print("Loading checkpoint.")
    checkpoint = torch.load("/orange/yonghui.wu/sgao1/llm_pruning_test.pth.tar", map_location=torch.device('cpu'))

    print("Initializing LLaMA 2-7B model.")
    api_token = 'your_hf_api_token'
    model_cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=api_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        attn_implementation="sdpa",
        token=api_token
    ).cuda()
    model.resize_token_embeddings(len(tokenizer))

    print("Loading state dict from checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print("Getting current mask vector.")
    cur_mask_vec = checkpoint["mask_vec"].to("cuda")
    masks = transform_output_layer_uniform(cur_mask_vec)

    # Include weight mask observation parts
    observe_weight_masks(model, model_cfg, masks)

    return model, tokenizer, masks

def observe_weight_masks(model, model_cfg, masks):
    # Check weights of the first layer's MLP
    print("Checking weights of the first layer's MLP:")
    print("gate_proj.weight:")
    print(model.model.layers[0].mlp.gate_proj.weight)
    print("up_proj.weight:")
    print(model.model.layers[0].mlp.up_proj.weight)
    print("down_proj.weight:")
    print(model.model.layers[0].mlp.down_proj.weight)

    # View current pruning pattern
    print("Viewing current pruning pattern.")
    attn_k_mask = masks[:32]
    attn_v_mask = masks[32:64]
    attn_out_mask = masks[-2]
    attn_k_pruning_dim = [(1 - inv_mask).sum(dim=1) for inv_mask in attn_k_mask]
    attn_v_pruning_dim = [(1 - inv_mask).sum(dim=1) for inv_mask in attn_v_mask]
    print(f"attn_k_pruning_pattern: {attn_k_pruning_dim}")
    print(f"attn_v_pruning_pattern: {attn_v_pruning_dim}")

    # Debugging for Group Lasso Weight Projection
    print("Viewing pruning patterns for each layer.")
    for layer_idx in range(32):
        layer_wise_masks = [individual_mask[layer_idx, :] for individual_mask in masks]
        mlp_up_mask = layer_wise_masks[-1]
        print(f"Layer {layer_idx}:")
        print(f"  mlp_up_mask_shape: {mlp_up_mask.size()}")
        mlp_up_mask_ratio = (1 - mlp_up_mask).sum() / mlp_up_mask.numel()
        print(f"  mlp_up_mask_ratio: {mlp_up_mask_ratio}")

    # Validate Group Lasso regularization
    print("Validating Group Lasso regularization.")
    gl_loss_module = Group_Lasso_regularization(
        args=None,
        target_llm_cfg=model_cfg,
        prunable_structure=None,
        fsdp_scaler=None
    )
    gl_loss_module.debug_purpose_compute(
        target_llm=model,
        pruning_masks=masks,
        epoch=None
    )

def evaluate_model_on_dataset(model, tokenizer, masks, dataset_name):
    if dataset_name.lower() == 'pubmedqa':
        dataset = load_dataset(
            "json",
            data_files="nlp_dataset_collections/PubMedQA/pubMedQA_test.jsonl"
        )["train"]
        evaluate_pubmedqa(model, tokenizer, masks, dataset)
    elif dataset_name.lower() == 'mednli':
        dataset = load_dataset(
            "json",
            data_files="nlp_dataset_collections/medNLI/mli_test_v1.jsonl"
        ).remove_columns(
            ["pairID", "sentence1_parse", "sentence1_binary_parse", "sentence2_parse", "sentence2_binary_parse"]
        )["train"]
        evaluate_mednli(model, tokenizer, masks, dataset)
    else:
        print(f"Dataset '{dataset_name}' is not supported.")
        return

def evaluate_pubmedqa(model, tokenizer, masks, dataset):
    print("Evaluating on PubMedQA dataset...")
    true_labels = []
    pred_labels = []

    for i in range(len(dataset)):
        context = " ".join(dataset[i]['CONTEXTS'])
        question = dataset[i]['QUESTION']
        gold_label = dataset[i]['final_decision'].lower()

        input_text = (
            f"The abstract of a biomedical research article is '{context}'. "
            f"Here comes a question '{question}', and please answer the question with 'yes', 'no', or 'maybe'. "
            f"The answer is '"
        )

        prediction = generate_predictions(model, tokenizer, masks, input_text)

        # Map prediction to one of the labels
        prediction = prediction.lower()
        if "yes" in prediction:
            prediction = 'yes'
        elif 'maybe' in prediction or 'ma' in prediction:
            prediction = 'maybe'
        elif 'no' in prediction:
            prediction = 'no'
        else:
            prediction = 'unknown'  # For unexpected predictions

        true_labels.append(gold_label)
        pred_labels.append(prediction)

        print(f"Sample {i+1}/{len(dataset)} | Gold: {gold_label} | Prediction: {prediction}")

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=['yes', 'no', 'maybe'], average=None, zero_division=0
    )

    # Calculate macro-F1 score
    macro_f1 = f1.mean()

    # Print per-class metrics
    for i, label in enumerate(['yes', 'no', 'maybe']):
        print(f"Class '{label}': Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {f1[i]:.4f}, Support: {support[i]}")

    print(f"\nMacro-F1 Score: {macro_f1:.4f}")



def evaluate_mednli(model, tokenizer, masks, dataset):
    print("Evaluating on MedNLI dataset...")
    acc_count_base = 0
    acc_count_masked = 0

    for i in range(len(dataset)):
        sentence1 = dataset[i]["sentence1"]
        sentence2 = dataset[i]["sentence2"]
        gold_label = dataset[i]["gold_label"]

        input_text = (
            f"Premise: '{sentence1}'\n"
            f"Hypothesis: '{sentence2}'\n"
            f"Based on the premise, is the hypothesis 'entailment', 'contradiction', or 'neutral'? The answer is '"
        )

        prediction_base, prediction_masked = generate_predictions(model, tokenizer, masks, input_text)

        if prediction_base == gold_label:
            acc_count_base += 1
        if prediction_masked == gold_label:
            acc_count_masked += 1

        print(f"Sample {i+1}/{len(dataset)} | Gold: {gold_label} | Base Prediction: {prediction_base} | Masked Prediction: {prediction_masked}")

    print(f"Base Model Accuracy: {acc_count_base / len(dataset) * 100:.2f}%")
    print(f"Masked Model Accuracy: {acc_count_masked / len(dataset) * 100:.2f}%")

def generate_predictions(model, tokenizer, input_text):
    generated_text = input_text

    model_inputs = tokenizer([generated_text], return_tensors="pt").to("cuda")

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Base model prediction
        model_output = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            return_dict=True
        )

    logits = model_output.logits
    next_token_logits = logits[:, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get next token predictions
    next_token_id = torch.argmax(probabilities, dim=-1)
    next_token = tokenizer.decode(next_token_id[0])

    return next_token 


if __name__ == "__main__":
    model, tokenizer, masks = initialize_model_and_tokenizer()
    dataset_name = input("Enter the dataset to evaluate (PubMedQA/MedNLI): ")
    evaluate_model_on_dataset(model, tokenizer, masks, dataset_name)