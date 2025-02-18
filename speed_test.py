import torch
import time

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# 生成 4096 维输入 tensor
input_tensor = torch.randn(4096, dtype=torch.float32, device=device)

# 选择要保留的元素数量
num_selected = 1500

# 1. **完全随机索引**（最坏情况）
random_indices = torch.randperm(4096, device=device)[:num_selected]

# 2. **低连续索引**（3-5 个连续块后跳跃）
low_continuity_indices = []
i = 0
while len(low_continuity_indices) < num_selected:
    length = torch.randint(3, 6, (1,), device=device).item()  # 连续 3-5 个
    low_continuity_indices.extend(range(i, min(i + length, 4096)))
    i += torch.randint(6, 10, (1,), device=device).item()  # 跳跃 6-10 个
low_continuity_indices = torch.tensor(low_continuity_indices[:num_selected], device=device)

# 3. **中等连续索引**（10-20 个连续块后跳跃）
medium_continuity_indices = []
i = 0
while len(medium_continuity_indices) < num_selected:
    length = torch.randint(10, 21, (1,), device=device).item()  # 连续 10-20 个
    medium_continuity_indices.extend(range(i, min(i + length, 4096)))
    i += torch.randint(20, 30, (1,), device=device).item()  # 跳跃 20-30 个
medium_continuity_indices = torch.tensor(medium_continuity_indices[:num_selected], device=device)

# 4. **高连续索引**（50-100 个连续块后跳跃）
high_continuity_indices = []
i = 0
while len(high_continuity_indices) < num_selected:
    length = torch.randint(50, 101, (1,), device=device).item()  # 连续 50-100 个
    high_continuity_indices.extend(range(i, min(i + length, 4096)))
    i += torch.randint(100, 150, (1,), device=device).item()  # 跳跃 100-150 个
high_continuity_indices = torch.tensor(high_continuity_indices[:num_selected], device=device)

# 5. **完全连续索引**（最优情况）
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
time_low_continuity = benchmark_index_select(low_continuity_indices, input_tensor)
time_medium_continuity = benchmark_index_select(medium_continuity_indices, input_tensor)
time_high_continuity = benchmark_index_select(high_continuity_indices, input_tensor)
time_continuous = benchmark_index_select(continuous_indices, input_tensor)

# **打印结果**
print(f"Device: {device}")
print(f"Average Execution Time (Random Indices) over 100 runs: {time_random:.6f} seconds")
print(f"Average Execution Time (Low Continuity Indices) over 100 runs: {time_low_continuity:.6f} seconds")
print(f"Average Execution Time (Medium Continuity Indices) over 100 runs: {time_medium_continuity:.6f} seconds")
print(f"Average Execution Time (High Continuity Indices) over 100 runs: {time_high_continuity:.6f} seconds")
print(f"Average Execution Time (Continuous Indices) over 100 runs: {time_continuous:.6f} seconds")

# **计算加速比**
print(f"Speedup (Continuous vs Random): {time_random / time_continuous:.2f}x")
print(f"Speedup (High Continuity vs Random): {time_random / time_high_continuity:.2f}x")
print(f"Speedup (Medium Continuity vs Random): {time_random / time_medium_continuity:.2f}x")
print(f"Speedup (Low Continuity vs Random): {time_random / time_low_continuity:.2f}x")