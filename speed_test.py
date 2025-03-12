import torch
import time

# 设备选择（CUDA 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 矩阵大小
N = 4096  # 4096 x 4096 矩阵

# ------------------- 1️⃣ 生成 2:4 稀疏性掩码 ------------------- #
def generate_2_4_sparsity_mask(rows, cols):
    """生成符合 2:4 稀疏性结构的掩码"""
    mask = torch.zeros(rows, cols, device=device)
    for row in range(rows):
        nonzero_cols = torch.randperm(cols, device=device)[:cols // 2]  # 每 4 选 2
        mask[row, nonzero_cols] = 1
    return mask

# 生成 2:4 掩码
sparsity_mask = generate_2_4_sparsity_mask(N, N)

# 创建密集矩阵并应用 2:4 掩码（数据类型为 `bfloat16`）
dense_matrix = torch.randn(N, N, device=device, dtype=torch.bfloat16)
sparse_matrix = (dense_matrix * sparsity_mask).to(torch.bfloat16)  # 确保 `bf16` 计算

# ------------------- 2️⃣ 转换为 PyTorch 稀疏格式 ------------------- #
sparse_coo = sparse_matrix.to_sparse()
sparse_csr = sparse_matrix.to_sparse_csr()

print(f"✅ COO 格式: {sparse_coo}")
print(f"✅ CSR 格式: {sparse_csr}")

# ------------------- 3️⃣ 生成随机输入向量 ------------------- #
input_vector = torch.randn(N, 4, device=device, dtype=torch.bfloat16)

# ------------------- 4️⃣ 预热（Warm-up） ------------------- #
print("🔥 预热中...")
for _ in range(10):  # 预热 10 次，避免冷启动
    _ = torch.matmul(sparse_matrix, input_vector)
    _ = torch.sparse.mm(sparse_coo, input_vector)
    _ = torch.sparse.mm(sparse_csr, input_vector)
torch.cuda.synchronize()

# ------------------- 5️⃣ 进行矩阵乘法计算 ------------------- #
def benchmark(func, desc):
    """通用基准测试函数"""
    torch.cuda.synchronize()
    start_time = time.time()
    result = func()
    torch.cuda.synchronize()
    duration = time.time() - start_time
    print(f"🚀 {desc} 计算时间: {duration:.6f} 秒")
    return result, duration

# 🔥 Dense 计算 (BF16)
dense_result, dense_time = benchmark(lambda: torch.matmul(sparse_matrix, input_vector), "Dense BF16")

# 🚀 Sparse COO 计算 (BF16)
sparse_result_coo, sparse_time_coo = benchmark(lambda: torch.sparse.mm(sparse_coo, input_vector), "Sparse COO BF16")

# 🚀 Sparse CSR 计算 (BF16)
sparse_result_csr, sparse_time_csr = benchmark(lambda: torch.sparse.mm(sparse_csr, input_vector), "Sparse CSR BF16")

# ------------------- 6️⃣ 结果验证 ------------------- #
print("✅ 结果相似度检查 (COO vs Dense):", torch.allclose(dense_result, sparse_result_coo, atol=1e-2))
print("✅ 结果相似度检查 (CSR vs Dense):", torch.allclose(dense_result, sparse_result_csr, atol=1e-2))

# ------------------- 7️⃣ 显示加速比 ------------------- #
print(f"\n💡 稀疏矩阵加速比 (COO vs Dense): {dense_time / sparse_time_coo:.2f}x")
print(f"💡 稀疏矩阵加速比 (CSR vs Dense): {dense_time / sparse_time_csr:.2f}x")