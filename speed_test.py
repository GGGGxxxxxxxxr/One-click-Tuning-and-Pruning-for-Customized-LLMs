import torch
import time

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# 生成 1M 维输入 tensor
input_tensor = torch.randn(10**6, dtype=torch.float32, device=device)

# 选择要保留的元素数量
num_selected = 100000  # 选取 10 万个索引

# 1. **完全随机索引**（最坏情况）
random_indices = torch.randperm(10**6, device=device)[:num_selected]

# **函数：生成不同连续性的索引**
def generate_continuous_indices(num_selected, block_size, jump_size, max_dim):
    """生成逐步增加连续性的索引"""
    indices = []
    i = 0
    while len(indices) < num_selected:
        indices.extend(range(i, min(i + block_size, max_dim)))  # 选取 block_size 个连续索引
        i += jump_size  # 进行跳跃
    return torch.tensor(indices[:num_selected], device=device)

# 2. **低连续索引**（3 连续 / 10 跳跃）
low_continuity_indices = generate_continuous_indices(num_selected, block_size=3, jump_size=10, max_dim=10**6)

# 3. **中等连续索引**（20 连续 / 50 跳跃）
medium_continuity_indices = generate_continuous_indices(num_selected, block_size=20, jump_size=50, max_dim=10**6)

# 4. **高连续索引**（100 连续 / 200 跳跃）
high_continuity_indices = generate_continuous_indices(num_selected, block_size=100, jump_size=200, max_dim=10**6)

# 5. **完全连续索引**（最优情况）
continuous_indices = torch.arange(num_selected, device=device)

# **性能测试函数**
def benchmark_index_select(indices, input_tensor, num_runs=50):
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
time_low_continuity = benchmark_index_select(low_continuity_indices, input_tensor)
time_medium_continuity = benchmark_index_select(medium_continuity_indices, input_tensor)
time_high_continuity = benchmark_index_select(high_continuity_indices, input_tensor)
time_continuous = benchmark_index_select(continuous_indices, input_tensor)

# **打印结果**
print(f"Device: {device}")
print(f"Average Execution Time (Random Indices) over 50 runs: {time_random:.6f} seconds")
print(f"Average Execution Time (Low Continuity Indices) over 50 runs: {time_low_continuity:.6f} seconds")
print(f"Average Execution Time (Medium Continuity Indices) over 50 runs: {time_medium_continuity:.6f} seconds")
print(f"Average Execution Time (High Continuity Indices) over 50 runs: {time_high_continuity:.6f} seconds")
print(f"Average Execution Time (Continuous Indices) over 50 runs: {time_continuous:.6f} seconds")

# **计算加速比**
print(f"Speedup (Continuous vs Random): {time_random / time_continuous:.2f}x")
print(f"Speedup (High Continuity vs Random): {time_random / time_high_continuity:.2f}x")
print(f"Speedup (Medium Continuity vs Random): {time_random / time_medium_continuity:.2f}x")
print(f"Speedup (Low Continuity vs Random): {time_random / time_low_continuity:.2f}x")