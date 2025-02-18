import torch
import time

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 生成 4096 维输入 tensor
input_tensor = torch.randn(4096, dtype=torch.float32, device=device)

# 2. 生成不同类型的索引
num_selected = 1500

# 方式 1: **完全随机索引**
random_indices = torch.randperm(4096, device=device)[:num_selected]

# 方式 2: **部分连续索引（分块取 50 个）**
semi_continuous_indices = torch.cat([torch.arange(i, i + 50, device=device) for i in range(0, num_selected * 2, 100)])[:num_selected]

# 方式 3: **完全连续索引**
continuous_indices = torch.arange(num_selected, device=device)

# **性能测试函数**
def benchmark_index_select(indices, input_tensor, num_runs=100):
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if device == "cuda" else None  # 确保 GPU 计算完成
        start_time = time.time()

        # 进行 index_select 操作
        selected_tensor = torch.index_select(input_tensor, 0, indices)

        torch.cuda.synchronize() if device == "cuda" else None  # 确保 GPU 计算完成
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    return sum(times) / num_runs  # 返回平均时间

# **运行测试**
time_random = benchmark_index_select(random_indices, input_tensor)
time_semi_continuous = benchmark_index_select(semi_continuous_indices, input_tensor)
time_continuous = benchmark_index_select(continuous_indices, input_tensor)

# **打印结果**
print(f"Device: {device}")
print(f"Average Execution Time (Random Indices) over 100 runs: {time_random:.6f} seconds")
print(f"Average Execution Time (Semi-Continuous Indices) over 100 runs: {time_semi_continuous:.6f} seconds")
print(f"Average Execution Time (Continuous Indices) over 100 runs: {time_continuous:.6f} seconds")

# **计算加速比**
print(f"Speedup (Continuous vs Random): {time_random / time_continuous:.2f}x")
print(f"Speedup (Semi-Continuous vs Random): {time_random / time_semi_continuous:.2f}x")