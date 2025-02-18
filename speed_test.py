import torch
import torch.nn as nn
import time

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 生成输入 tensor (4096 维)
input_tensor = torch.randn(4096, dtype=torch.float32, device=device)

# 2. 随机采样 1500 个索引
num_selected = 1500
indices = torch.randperm(4096, device=device)[:num_selected]  # 生成 1500 个唯一随机索引

# 3. 使用 index_select 选择索引后的 tensor
selected_tensor = torch.index_select(input_tensor, 0, indices)  # 变成 (1500,)

# 4. 定义 MLP (Up-projection -> Down-projection)
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

# 5. 初始化 MLP 并移动到 GPU/CPU
mlp = MLP().to(device)

# 速度测试：多次运行取平均值
num_runs = 100  # 运行次数
times = []

for _ in range(num_runs):
    torch.cuda.synchronize() if device == "cuda" else None  # 确保 GPU 计算完成
    start_time = time.time()

    # --- 算子执行部分 ---
    selected_tensor = torch.index_select(input_tensor, 0, indices)  # 选取索引
    updated_tensor = mlp(selected_tensor)  # MLP 投影计算
    output_tensor = input_tensor.clone()  # 复制原始输入
    output_tensor.index_add_(0, indices, updated_tensor)  # 加回

    torch.cuda.synchronize() if device == "cuda" else None  # 确保 GPU 计算完成
    end_time = time.time()
    
    times.append(end_time - start_time)

# 计算平均时间
avg_time = sum(times) / num_runs

# 输出结果
print(f"Device: {device}")
print(f"Average Execution Time over {num_runs} runs: {avg_time:.6f} seconds")