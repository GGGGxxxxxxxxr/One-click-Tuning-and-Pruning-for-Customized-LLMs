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
low_continuity_indices = torch.cat([
    torch.arange(i, i + torch.randint(3, 6, (1,), device=device).item(), device=device)
    for i in range(0, num_selected * 3, 10)
])[:num_selected]

# 3. **中等连续索引**（10-20 个连续块后跳跃）
medium_continuity_indices = torch.cat([
    torch.arange(i, i + torch.randint(10, 21, (1,), device=device).item(), device=device)
    for i in range(0, num_selected * 3, 50)
])[:num_selected]

# 4. **高连续索引**（50-100 个连续块后跳跃）
high_continuity_indices = torch.cat([
    torch.arange(i, i + torch.randint(50, 101, (1,), device=device).item(), device=device)
    for i in range(0, num_selected * 3, 200)
])[:num_selected]

# 5. **完全连续索引**（最优情况）
continuous_indices = torch.arange(num_selected, device=device)

# **性能测试函数**
def benchmark_index_select(indices, input_tensor, num_runs=500):
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