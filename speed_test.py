import torch
import torch.nn as nn
import time

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 生成输入 tensor (4096 维)
input_tensor = torch.randn(4096, dtype=torch.float32, device=device)

# 2. 预定义 1500 维索引（随机索引 vs. 切片）
num_selected = 1500
indices_random = torch.randperm(4096, device=device)[:num_selected]  # 随机索引
indices_slice = torch.arange(num_selected, device=device)  # 直接切片

# 3. 定义 MLP (Up-projection -> Down-projection)
class MLP(nn.Module):
    def __init__(self, in_dim=1500, hidden_dim=2048, out_dim=1500):
        super(MLP, self).__init__()
        self.up_proj = nn.Linear(in_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.up_proj(x))  # Up-projection
        x = self.down_proj(x)  # Down-projection
        return x

# 4. 初始化 MLP 并移动到 GPU/CPU
mlp = MLP().to(device)

# 速度测试参数
num_runs = 100  # 运行次数

# 5. **测试方法 1：使用随机索引 (`index_select`)**
times_random = []
for _ in range(num_runs):
    torch.cuda.synchronize() if device == "cuda" else None  # 确保 GPU 计算完成
    start_time = time.time()

    # --- 算子执行部分 ---
    selected_tensor = torch.index_select(input_tensor, 0, indices_random)  # 选取索引
    updated_tensor = mlp(selected_tensor)  # MLP 投影计算
    output_tensor = input_tensor.clone()  # 复制原始输入
    output_tensor.index_add_(0, indices_random, updated_tensor)  # 加回

    torch.cuda.synchronize() if device == "cuda" else None  # 确保 GPU 计算完成
    end_time = time.time()
    
    times_random.append(end_time - start_time)

avg_time_random = sum(times_random) / num_runs

# 6. **测试方法 2：使用切片 (`slice`)**
times_slice = []
for _ in range(num_runs):
    torch.cuda.synchronize() if device == "cuda" else None  # 确保 GPU 计算完成
    start_time = time.time()

    # --- 算子执行部分 ---
    selected_tensor = input_tensor[:num_selected]  # 直接切片
    updated_tensor = mlp(selected_tensor)  # MLP 投影计算
    output_tensor = input_tensor.clone()  # 复制原始输入
    output_tensor[:num_selected] += updated_tensor  # 直接加回

    torch.cuda.synchronize() if device == "cuda" else None  # 确保 GPU 计算完成
    end_time = time.time()
    
    times_slice.append(end_time - start_time)

avg_time_slice = sum(times_slice) / num_runs

# **打印最终对比结果**
print(f"Device: {device}")
print(f"Average Execution Time (Random Index Select) over {num_runs} runs: {avg_time_random:.6f} seconds")
print(f"Average Execution Time (Slice First 1500) over {num_runs} runs: {avg_time_slice:.6f} seconds")
print(f"Speedup Factor (Slice vs. Random Index): {avg_time_random / avg_time_slice:.2f}x")