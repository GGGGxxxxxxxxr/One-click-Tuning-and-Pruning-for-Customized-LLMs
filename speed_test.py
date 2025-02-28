import torch
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Dummy Model (Baseline)
class BaselineModel(torch.nn.Module):
    def __init__(self, hidden_dim=4096, intermediate_dim=11008):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(hidden_dim, dtype=torch.bfloat16, device=device)
        self.gate = torch.nn.Linear(hidden_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.up = torch.nn.Linear(hidden_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.down = torch.nn.Linear(intermediate_dim, hidden_dim, device=device, dtype=torch.bfloat16)

    def forward(self, hidden_states):
        residual = hidden_states.clone()
        hidden_states = self.layernorm(hidden_states)
        gate_out = torch.sigmoid(self.gate(hidden_states))  
        up_out = torch.relu(self.up(hidden_states))         
        mlp_out = self.down(gate_out * up_out)              
        return residual + mlp_out

# Initialize model
baseline_model = BaselineModel(hidden_dim=4096, intermediate_dim=11008).to(device)
baseline_model.eval()

# Dummy Input
batch_size = 1
seq_len = 128
hidden_dim = 4096
input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)

# 🔥 Warm-up Phase (Avoid Cold Start Effects)
print("🔥 Running warm-up iterations...")
for _ in range(20):
    _ = baseline_model(input_tensor)
torch.cuda.synchronize()

# 🚀 PyTorch Profiler with Proper Scheduling
print("\nProfiling Baseline Model...\n")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    schedule=profiler.schedule(wait=5, warmup=10, active=20), 
    record_shapes=True, 
    with_stack=True
) as prof:
    for _ in range(50):  # Run enough iterations for schedule() to activate
        _ = baseline_model(input_tensor)
        prof.step()  # ✅ Ensure the profiler progresses

# 📊 Print Results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))