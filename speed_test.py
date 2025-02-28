import torch
import time

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Parameters
hidden_dim = 4096
pruned_dim = 4096
intermediate_dim = 11008
batch_size = 1
seq_len = 128

# Generate random input tensors
input_fp32 = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
input_bf16 = input_fp32.to(dtype=torch.bfloat16)

# Generate random pruning indices
s3_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)
s5_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)

# Define CUDA Events for Precise Timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Function to Profile a Single Operation
def profile_operation(operation, input_tensor, num_runs=10000):
    torch.cuda.synchronize()  # Ensure all previous operations are completed

    # Warm-up to avoid cold start effects
    for _ in range(10):
        _ = operation()

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_runs):
        _ = operation()

    end_event.record()
    torch.cuda.synchronize()  # Ensure all operations are completed

    return start_event.elapsed_time(end_event) / num_runs  # Average time in milliseconds


# Measure FP32 `index_select`
index_select_fp32 = lambda: torch.index_select(input_fp32, -1, s3_index)
index_select_time_fp32 = profile_operation(index_select_fp32, input_fp32)

# Measure BF16 `index_select`
index_select_bf16 = lambda: torch.index_select(input_bf16, -1, s3_index)
index_select_time_bf16 = profile_operation(index_select_bf16, input_bf16)


# Measure FP32 `index_add`
residual_fp32 = input_fp32.clone()
index_add_fp32 = lambda: residual_fp32.index_add(-1, s5_index, input_fp32)
index_add_time_fp32 = profile_operation(index_add_fp32, input_fp32)

# Measure BF16 `index_add`
residual_bf16 = input_bf16.clone()
index_add_bf16 = lambda: residual_bf16.index_add(-1, s5_index, input_bf16)
index_add_time_bf16 = profile_operation(index_add_bf16, input_bf16)


# Print Results
print("\n=== Performance Comparison (FP32 vs. BF16) ===")
print(f"Index Select (FP32): {index_select_time_fp32:.6f} ms")
print(f"Index Select (BF16): {index_select_time_bf16:.6f} ms")
print(f"Speedup (Index Select): {index_select_time_fp32 / index_select_time_bf16:.2f}x\n")

print(f"Index Add (FP32): {index_add_time_fp32:.6f} ms")
print(f"Index Add (BF16): {index_add_time_bf16:.6f} ms")
print(f"Speedup (Index Add): {index_add_time_fp32 / index_add_time_bf16:.2f}x")