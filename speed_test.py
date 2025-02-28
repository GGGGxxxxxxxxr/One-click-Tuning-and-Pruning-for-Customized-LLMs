import torch
import time

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Pruned Model with Index Selection and Addition
class PrunedModel(torch.nn.Module):
    def __init__(self, hidden_dim=4096, pruned_dim=4096, intermediate_dim=11008):
        super().__init__()
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim)
        
        # Gated MLP components
        self.gate = torch.nn.Linear(pruned_dim, intermediate_dim)
        self.up = torch.nn.Linear(pruned_dim, intermediate_dim)
        self.down = torch.nn.Linear(intermediate_dim, pruned_dim)

        # Simulated pruning indices
        self.s3_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)
        self.s5_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)

    def forward(self, hidden_states):
        residual = hidden_states.clone()

        # Apply pruning before MLP
        hidden_states = self.post_attention_layernorm(hidden_states)                               
        hidden_states = torch.index_select(hidden_states, -1, self.s3_index)                         

        # Compute Gated MLP
        gate_out = torch.sigmoid(self.gate(hidden_states))  
        up_out = torch.relu(self.up(hidden_states))         
        mlp_out = self.down(gate_out * up_out)              

        # Apply pruning before adding back to residual
        hidden_states = residual.index_add(-1, self.s5_index, mlp_out.contiguous())            

        return hidden_states


# Define Baseline Model (Without Index Selection and Addition)
class BaselineModel(torch.nn.Module):
    def __init__(self, hidden_dim=4096, intermediate_dim=11008):
        super().__init__()
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim)
        
        # Gated MLP components
        self.gate = torch.nn.Linear(hidden_dim, intermediate_dim)
        self.up = torch.nn.Linear(hidden_dim, intermediate_dim)
        self.down = torch.nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, hidden_states):
        residual = hidden_states.clone()

        # Apply LayerNorm
        hidden_states = self.post_attention_layernorm(hidden_states)                               

        # Compute Gated MLP
        gate_out = torch.sigmoid(self.gate(hidden_states))  
        up_out = torch.relu(self.up(hidden_states))         
        mlp_out = self.down(gate_out * up_out)              

        # Direct residual connection without pruning operations
        hidden_states = residual + mlp_out            

        return hidden_states


# Initialize models
pruned_model = PrunedModel(hidden_dim=4096, pruned_dim=4096, intermediate_dim=11008).to(device)
baseline_model = BaselineModel(hidden_dim=4096, intermediate_dim=11008).to(device)

pruned_model.eval()
baseline_model.eval()

# Generate random input tensor
batch_size = 1
seq_len = 128
hidden_dim = 4096
input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)

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

# Print results
print(f"Pruned Model Average Inference Time: {pruned_latency:.2f} ms per forward pass")
print(f"Baseline Model Average Inference Time: {baseline_latency:.2f} ms per forward pass")
print(f"Overhead due to index operations: {pruned_latency - baseline_latency:.2f} ms")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy input
hidden_dim = 4096
pruned_dim = 4096
batch_size = 1
seq_len = 128

input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
s3_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)

# CUDA Event Timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start_event.record()

selected_tensor = torch.index_select(input_tensor, -1, s3_index)

end_event.record()
torch.cuda.synchronize()

elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
print(f"index_select execution time: {elapsed_time:.3f} ms")