import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
SparseSemiStructuredTensor._FORCE_CUTLASS = True

# mask Linear weight to be 2:4 sparse
mask = torch.Tensor([0, 0, 1, 1]).tile((4096, 1024)).cuda().bool().to(torch.bfloat16)
linear = torch.nn.Linear(4096, 4096).bfloat16().cuda().eval()
linear.weight = torch.nn.Parameter(mask * linear.weight)

x = torch.rand(1280, 4096).bfloat16().cuda()

with torch.inference_mode():
    with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,  # 记录 Tensor 形状
    profile_memory=True,  # 记录显存使用情况
    with_stack=True  # 记录调用栈
    ) as prof:
        for _ in range(10):  # 运行多次以获取稳定数据
            dense_output = linear(x)

    # 🔍 打印 Profiling 结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    dense_output = linear(x)
    dense_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # accelerate via SparseSemiStructuredTensor
    linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))

    with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,  # 记录 Tensor 形状
    profile_memory=True,  # 记录显存使用情况
    with_stack=True  # 记录调用栈
    ) as prof:
        for _ in range(10):  # 运行多次以获取稳定数据
            sparse_output = linear(x)
    # 🔍 打印 Profiling 结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    sparse_output = linear(x)
    sparse_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # sparse and dense matmul are numerically equivalent
    assert torch.allclose(sparse_output, dense_output, atol=1e-2)
    print(f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup: {(dense_t / sparse_t):.3f}x")