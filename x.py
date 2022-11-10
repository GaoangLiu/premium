import torch
import torch.nn.functional as F

source = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(source.shape)

print(source)
print('after')
source_pad = F.pad(source, pad=(0,20 - source.shape[0]))
print(source_pad)


