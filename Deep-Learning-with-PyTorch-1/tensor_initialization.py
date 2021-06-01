import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)  # 3 rows and 1 column
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)


# Other common initialization methods
x_empty = torch.empty(size=(3, 3))
x_zeros = torch.zeros(size=(3, 3))
x_rand = torch.rand(size=(3, 3))
x_ones = torch.ones(size=(3, 3))
x_identity = torch.eye(5, 5)
x_range = torch.arange(start=0, end=5, step=1)  # end is not inclusive
x_linspace = torch.linspace(start=0.1, end=1, steps=12)
x_norm = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x_unif = torch.empty(size=(1, 5)).uniform_(0, 1)
x_diag = torch.diag(torch.ones(3))


# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())


# Array to Tensor conversion and vice versa
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
print(tensor)
np_array_back = tensor.numpy()
print(np_array_back)