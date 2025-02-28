import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = 4096
batch_size = 1
seq_len = 128

# Random BF16 input tensor
input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
index_tensor = torch.randint(0, hidden_dim, (hidden_dim,), device=device)

result = input_tensor.index_add(-1, index_tensor, input_tensor)

# Profile `index_add`
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("BF16 index_add"):
        result = input_tensor.index_add(-1, index_tensor, input_tensor)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))