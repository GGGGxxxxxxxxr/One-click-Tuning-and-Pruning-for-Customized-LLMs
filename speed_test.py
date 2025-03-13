import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
SparseSemiStructuredTensor._FORCE_CUTLASS = False # Flase will use CUSPARSELt which is faster than CUTLASS

# mask Linear weight to be 2:4 sparse
mask = torch.Tensor([0, 0, 1, 1]).tile((3072, 768)).cuda().bool().to(torch.bfloat16)
linear = torch.nn.Linear(3072, 3072).bfloat16().cuda().eval()
linear.weight = torch.nn.Parameter(mask * linear.weight)

x = torch.rand(1, 3072).bfloat16().cuda()

with torch.inference_mode():
    

    dense_output = linear(x)
    dense_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # accelerate via SparseSemiStructuredTensor
    linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))


    sparse_output = linear(x)
    sparse_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # sparse and dense matmul are numerically equivalent
    assert torch.allclose(sparse_output, dense_output, atol=1e-2)
    print(f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup: {(dense_t / sparse_t):.3f}x")