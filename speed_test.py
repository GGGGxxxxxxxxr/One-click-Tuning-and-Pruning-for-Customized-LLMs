import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
import torch.profiler as profiler

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Pruned Model with Index Selection and Addition
class PrunedModel(torch.nn.Module):
    def __init__(self, hidden_dim=4096, pruned_dim=4096, intermediate_dim=11008):
        super().__init__()
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, dtype=torch.bfloat16, device=device)
        
        # Gated MLP components
        self.gate = torch.nn.Linear(pruned_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.up = torch.nn.Linear(pruned_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.down = torch.nn.Linear(intermediate_dim, pruned_dim, device=device, dtype=torch.bfloat16)

        # Simulated pruning indices (on GPU)
        self.s3_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)
        self.s5_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)

    def forward(self, hidden_states):
        residual = hidden_states.clone()

        with record_function("LayerNorm"):
            hidden_states = self.post_attention_layernorm(hidden_states)

        with record_function("IndexSelect (s3)"):
            hidden_states = torch.index_select(hidden_states, -1, self.s3_index)

        with record_function("Gated MLP - Gate/Up"):
            gate_out = torch.sigmoid(self.gate(hidden_states))  
            up_out = torch.relu(self.up(hidden_states))         

        with record_function("Gated MLP - Down"):
            mlp_out = self.down(gate_out * up_out)              

        with record_function("IndexAdd (s5)"):
            hidden_states = residual.index_add(-1, self.s5_index, mlp_out.contiguous())            

        return hidden_states.to(dtype=torch.bfloat16)


# Define Baseline Model (Without Index Selection and Addition)
class BaselineModel(torch.nn.Module):
    def __init__(self, hidden_dim=4096, intermediate_dim=11008):
        super().__init__()
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, dtype=torch.bfloat16, device=device)
        
        # Gated MLP components
        self.gate = torch.nn.Linear(hidden_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.up = torch.nn.Linear(hidden_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.down = torch.nn.Linear(intermediate_dim, hidden_dim, device=device, dtype=torch.bfloat16)

    def forward(self, hidden_states):
        residual = hidden_states.clone()

        with record_function("LayerNorm"):
            hidden_states = self.post_attention_layernorm(hidden_states)

        with record_function("Gated MLP - Gate/Up"):
            gate_out = torch.sigmoid(self.gate(hidden_states))  
            up_out = torch.relu(self.up(hidden_states))         

        with record_function("Gated MLP - Down"):
            mlp_out = self.down(gate_out * up_out)              

        with record_function("Residual Connection"):
            hidden_states = residual + mlp_out            

        return hidden_states.to(dtype=torch.bfloat16)

# Initialize models
pruned_model = PrunedModel(hidden_dim=4096, pruned_dim=4096, intermediate_dim=11008).to(device)
baseline_model = BaselineModel(hidden_dim=4096, intermediate_dim=11008).to(device)

pruned_model.eval()
baseline_model.eval()

# Generate random input tensor in BF16
batch_size = 1
seq_len = 128
hidden_dim = 4096
input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)

# Function to measure inference time
def measure_latency(model, input_tensor, num_runs=10000):
    with torch.no_grad():
        for _ in range(10):  # Warm-up
            _ = model(input_tensor)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) / num_runs * 1000  # Convert to ms

# Measure latency for both models
pruned_latency = measure_latency(pruned_model, input_tensor)
baseline_latency = measure_latency(baseline_model, input_tensor)

# Profile both models
print("\nProfiling Pruned Model...\n")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    pruned_model(input_tensor)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1000))

print("\nProfiling Baseline Model...\n")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=profiler.schedule(wait=5, warmup=10, active=20), record_shapes=True) as prof:
    baseline_model(input_tensor)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1000))

# Print results
print(f"\nPruned Model Average Inference Time: {pruned_latency:.6f} ms per forward pass")
print(f"Baseline Model Average Inference Time: {baseline_latency:.6f} ms per forward pass")
print(f"Overhead due to index operations: {pruned_latency - baseline_latency:.6f} ms")