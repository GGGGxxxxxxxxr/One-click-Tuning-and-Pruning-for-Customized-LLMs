import torch
import time

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 🚀 Define Baseline Model ---------------------- #
class BaselineModel(torch.nn.Module):
    def __init__(self, hidden_dim=4096, intermediate_dim=11008):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(hidden_dim, dtype=torch.bfloat16, device=device)
        self.gate = torch.nn.Linear(hidden_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.up = torch.nn.Linear(hidden_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.down = torch.nn.Linear(intermediate_dim, hidden_dim, device=device, dtype=torch.bfloat16)

    def forward(self, hidden_states):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        residual = hidden_states.clone()
        hidden_states = self.layernorm(hidden_states)
        gate_out = torch.sigmoid(self.gate(hidden_states))  
        up_out = torch.relu(self.up(hidden_states))         
        mlp_out = self.down(gate_out * up_out)              
        hidden_states = residual + mlp_out
        end_event.record()

        torch.cuda.synchronize()
        return hidden_states, start_event.elapsed_time(end_event)

# ---------------------- ✂️ Define Pruned Model ---------------------- #
class PrunedModel(torch.nn.Module):
    def __init__(self, hidden_dim=4096, pruned_dim=2048, intermediate_dim=5004):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(hidden_dim, dtype=torch.bfloat16, device=device)
        self.gate = torch.nn.Linear(pruned_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.up = torch.nn.Linear(pruned_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.down = torch.nn.Linear(intermediate_dim, pruned_dim, device=device, dtype=torch.bfloat16)

        # Simulated pruning indices (GPU)
        self.s3_index = torch.sort(torch.randperm(hidden_dim, device=device)[:pruned_dim])[0]
        self.s5_index = torch.sort(torch.randperm(hidden_dim, device=device)[:pruned_dim])[0]

    def forward(self, hidden_states):
        residual = hidden_states.clone().contiguous()

        # 🚀 Start CUDA event for full forward pass
        start_forward = torch.cuda.Event(enable_timing=True)
        end_forward = torch.cuda.Event(enable_timing=True)

        # 🚀 Start CUDA event for `index_select`
        start_select = torch.cuda.Event(enable_timing=True)
        end_select = torch.cuda.Event(enable_timing=True)

        # 🚀 Start CUDA event for `index_add`
        start_add = torch.cuda.Event(enable_timing=True)
        end_add = torch.cuda.Event(enable_timing=True)

        start_forward.record()  # Start total forward timing

        start_select.record()
        hidden_states = self.layernorm(hidden_states)
        hidden_states = torch.index_select(hidden_states.contiguous(), -1, self.s3_index)
        end_select.record()  # End `index_select` timing

        # Compute Gated MLP
        gate_out = torch.sigmoid(self.gate(hidden_states))
        up_out = torch.relu(self.up(hidden_states))
        mlp_out = self.down(gate_out * up_out)

        start_add.record()
        hidden_states = residual.index_add(-1, self.s5_index, mlp_out.contiguous())
        end_add.record()  # End `index_add` timing

        end_forward.record()  # End total forward timing

        # Synchronize for accurate measurement
        torch.cuda.synchronize()
        forward_time = start_forward.elapsed_time(end_forward)
        index_select_time = start_select.elapsed_time(end_select)
        index_add_time = start_add.elapsed_time(end_add)

        return hidden_states.to(dtype=torch.bfloat16), forward_time, index_select_time, index_add_time

# ---------------------- 🎯 Initialize Models ---------------------- #
baseline_model = BaselineModel(hidden_dim=4096, intermediate_dim=11008).to(device)
pruned_model = PrunedModel(hidden_dim=4096, pruned_dim=2048, intermediate_dim=5004).to(device)

baseline_model.eval()
pruned_model.eval()

# Dummy Input (Both FP32 and BF16)
batch_size = 1
seq_len = 1
hidden_dim = 4096
input_tensor_fp32 = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
input_tensor_bf16 = input_tensor_fp32.to(dtype=torch.bfloat16)

# 🔥 Warm-up Phase (Avoid Cold Start Effects)
print("🔥 Running warm-up iterations...")
for _ in range(20):
    _, _ = baseline_model(input_tensor_bf16)
    _, _, _, _ = pruned_model(input_tensor_bf16)
torch.cuda.synchronize()

# 🚀 Measure Execution Time
def benchmark_model(model, input_tensor, model_name, num_runs=100):
    total_forward_time = 0
    total_index_select_time = 0
    total_index_add_time = 0

    print(f"\n⏳ Benchmarking {model_name}...\n")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_runs):
        if model_name == "Baseline Model":
            _, forward_time = model(input_tensor)
            total_forward_time += forward_time
        else:
            _, forward_time, select_time, add_time = model(input_tensor)
            total_forward_time += forward_time
            total_index_select_time += select_time
            total_index_add_time += add_time

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 📊 Compute Averages
    avg_forward_time = total_forward_time / num_runs
    avg_index_select_time = total_index_select_time / num_runs if model_name == "Pruned Model" else 0
    avg_index_add_time = total_index_add_time / num_runs if model_name == "Pruned Model" else 0
    total_runtime = (end_time - start_time) * 1000 / num_runs  # Convert to ms

    # 📢 Print Results
    print(f"\n🚀 Benchmark Results for {model_name} over {num_runs} runs:")
    print(f"🔥 Full Forward Pass: {avg_forward_time:.3f} ms")
    if model_name == "Pruned Model":
        print(f"🟢 index_select Time: {avg_index_select_time:.3f} ms")
        print(f"🔴 index_add Time: {avg_index_add_time:.3f} ms")
    print(f"⏱️  Total Runtime (Including Python Overhead): {total_runtime:.3f} ms\n")

# ---------------------- 🏎️ Run Benchmarks ---------------------- #
print("\n📌 Benchmarking with BF16:")
benchmark_model(baseline_model, input_tensor_bf16, "Baseline Model")
benchmark_model(pruned_model, input_tensor_bf16, "Pruned Model")

print("\n📌 Benchmarking with FP32:")
benchmark_model(baseline_model, input_tensor_fp32, "Baseline Model")
benchmark_model(pruned_model, input_tensor_fp32, "Pruned Model")