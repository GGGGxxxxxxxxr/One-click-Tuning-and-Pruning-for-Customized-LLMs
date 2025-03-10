import torch
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity

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
        residual = hidden_states.clone()
        hidden_states = self.layernorm(hidden_states)
        gate_out = torch.sigmoid(self.gate(hidden_states))  
        up_out = torch.relu(self.up(hidden_states))         
        mlp_out = self.down(gate_out * up_out)              
        residual = residual + mlp_out
        return residual

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

        # Apply index selection (simulated pruning)
        hidden_states = self.layernorm(hidden_states)                               
        hidden_states = torch.index_select(hidden_states.contiguous(), -1, self.s3_index)    
        #hidden_states = hidden_states[:,:,:2048]                     

        # Compute Gated MLP
        gate_out = torch.sigmoid(self.gate(hidden_states))  
        up_out = torch.relu(self.up(hidden_states))         
        mlp_out = self.down(gate_out * up_out)              

        # Apply index addition before adding back to residual
        hidden_states = residual.index_add(-1, self.s5_index, mlp_out.contiguous())     
        #residual[:,:,:2048] = residual[:,:,:2048] + mlp_out      

        return hidden_states.to(dtype=torch.bfloat16)

# ---------------------- 🎯 Initialize Models ---------------------- #
baseline_model = BaselineModel(hidden_dim=4096, intermediate_dim=11008).to(device)
pruned_model = PrunedModel(hidden_dim=4096, pruned_dim=2048, intermediate_dim=5004).to(device)


baseline_model.eval()
pruned_model.eval()

baseline_model = torch.compile(baseline_model)
pruned_model = torch.compile(pruned_model)

# Dummy Input
batch_size = 32
seq_len = 1
hidden_dim = 4096
input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)

# 🔥 Warm-up Phase (Avoid Cold Start Effects)
print("🔥 Running warm-up iterations...")
for _ in range(20):
    _ = baseline_model(input_tensor)
    _ = pruned_model(input_tensor)
torch.cuda.synchronize()

# 🚀 Profiling Function (NO schedule)
def profile_model(model, model_name):
    print(f"\n📊 Profiling {model_name}...\n")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    ) as prof:
        for _ in range(50):  # Run enough iterations to collect meaningful data
            _ = model(input_tensor)

    # 📊 Print Results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# ---------------------- 🏎️ Run Profiling ---------------------- #
profile_model(baseline_model, "Baseline Model")
profile_model(pruned_model, "Pruned Model")