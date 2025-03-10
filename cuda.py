
from torch.sparse import to_sparse_semi_structured
import torch
import time 

a = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).half().cuda()
b = torch.rand(64, 64).half().cuda()
c = torch.mm(a, b)
a_sparse = to_sparse_semi_structured(a)

start_time = time.time()
print(torch.mm(a_sparse, b))
end_time = time.time()

print(end_time - start_time)