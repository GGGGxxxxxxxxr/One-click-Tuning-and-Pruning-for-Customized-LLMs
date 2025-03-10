import torch
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity
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
        residual = hidden_states.clone()
        hidden_states = self.layernorm(hidden_states)
        gate_out = torch.sigmoid(self.gate(hidden_states))  
        up_out = torch.relu(self.up(hidden_states))         
        mlp_out = self.down(gate_out * up_out)              
        residual = residual + mlp_out
        return residual

# ---------------------- ✂️ Define Pruned Model ---------------------- #
class PrunedModel(torch.nn.Module):
    def __init__(self, hidden_dim=4096, pruned_dim=2048, intermediate_dim=5004, group_size=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(hidden_dim, dtype=torch.bfloat16, device=device)
        self.gate = torch.nn.Linear(pruned_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.up = torch.nn.Linear(pruned_dim, intermediate_dim, device=device, dtype=torch.bfloat16)
        self.down = torch.nn.Linear(intermediate_dim, pruned_dim, device=device, dtype=torch.bfloat16)

        # --- 🚀 Group-wise structured index selection ---
        def generate_groupwise_indices(hidden_dim, pruned_dim, group_size):
            indices = torch.cat([
                torch.arange(i, i + group_size, device=device)
                for i in range(0, hidden_dim, 2 * group_size)  # 选 group_size，跳过 group_size
            ])
            return indices[:pruned_dim]  # 截断到 pruned_dim

        self.s3_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)
        self.s5_index = torch.randint(0, hidden_dim, (pruned_dim,), device=device)

    def forward(self, hidden_states):
        residual = hidden_states.clone().contiguous()

        # --- 🏆 Apply group-wise index selection ---
        hidden_states = self.layernorm(hidden_states)                               
        hidden_states = torch.index_select(hidden_states.contiguous(), -1, self.s3_index)    

        # Compute Gated MLP
        gate_out = torch.sigmoid(self.gate(hidden_states))  
        up_out = torch.relu(self.up(hidden_states))         
        mlp_out = self.down(gate_out * up_out)              

        # --- 🏆 Apply group-wise index addition ---
        hidden_states = residual.index_add(-1, self.s5_index, mlp_out.contiguous())     

        return hidden_states.to(dtype=torch.bfloat16)

# ---------------------- 🎯 Initialize Models ---------------------- #
baseline_model = BaselineModel(hidden_dim=4096, intermediate_dim=11008).to(device)
pruned_model = PrunedModel(hidden_dim=4096, pruned_dim=2048, intermediate_dim=5004).to(device)


baseline_model.eval()
pruned_model.eval()

#baseline_model = torch.compile(baseline_model)
#pruned_model = torch.compile(pruned_model)

# Dummy Input
batch_size = 128
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
start_time = time.time()
profile_model(baseline_model, "Baseline Model")
end_time = time.time()
duration = end_time - start_time
print(duration)

start_time = time.time()
profile_model(pruned_model, "Pruned Model")
end_time = time.time()
duration = end_time - start_time
print(duration)