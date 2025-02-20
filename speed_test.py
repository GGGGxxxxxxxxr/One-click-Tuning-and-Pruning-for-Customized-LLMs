import torch
import time

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model with Custom Pruned Inference
class PrunedModel(torch.nn.Module):
    def __init__(self, hidden_dim=1024, mlp_dim=4096, pruned_dim=512):
        super().__init__()
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim)
        self.mlp = torch.nn.Linear(pruned_dim, hidden_dim)  # Simulating MLP block
        self.hidden_dim = hidden_dim

        # Simulated pruning indices (random for benchmarking)
        self.s3_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)
        self.s5_index = torch.randint(0, hidden_dim, (hidden_dim,), device=device)

    def forward(self, hidden_states):
        residual = hidden_states.clone()

        # [ATP_DISP]: Apply s3 before MLP_block
        hidden_states = self.post_attention_layernorm(hidden_states)                               
        hidden_states = torch.index_select(hidden_states, -1, self.s3_index)                         

        # [ATP_DISP]: Infuse s4, s5 into self.mlp.forward()        
        hidden_states = self.mlp(hidden_states)                                                 
        hidden_states = residual.index_add(-1, self.s5_index, hidden_states.contiguous())            

        return hidden_states

# Create the model and move it to GPU
model = PrunedModel(hidden_dim=1024, mlp_dim=4096, pruned_dim=512).to(device)
model.eval()

# Generate random input tensor
batch_size = 32
seq_len = 128
hidden_dim = 1024
input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)

# Warm-up CUDA (avoid cold-start delays)
with torch.no_grad():
    for _ in range(10):
        _ = model(input_tensor)

# Measure inference time
num_runs = 100
torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
    for _ in range(num_runs):
        _ = model(input_tensor)

torch.cuda.synchronize()
end_time = time.perf_counter()

# Compute average latency
avg_latency = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
print(f"Average Inference Time: {avg_latency:.2f} ms per forward pass")