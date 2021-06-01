import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([20, 30, 40])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
print(z2)

z3 = x + y
print(z3)

# subtraction
z4 = x - y
print(z4)

# Division
z5 = torch.true_divide(x, y) # Similar to broadcasting
print(z5)

t = torch.zeros(3)
print(t)
t.add_(x)  # function with underscores are inplace
print(t)

# Exponentiation
z6 = x.pow(2)
print(z6)

z7 = x ** 2
print(z7)

# Simple comparison
z8 = x > 0
print(z8)

z9 = x < 0

# Matrix multiply

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
print(x3)
x4 = x1.mm(x2)
print(x4)

# matrix exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

# Element wise multiplication
z10 = x * y
print(z10)

# dot product
z = torch.dot(x, y)
print(z)

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1,tensor2)
# print(out_bmm)


# Example of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2 # Vector is subtracted from all the rows
z = x1 ** x2
print(z)

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0) # we can do x.max(dim=0) too
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0) # torch expects the array to be floats
z = torch.eq(x, y)
z = torch.sort(y, dim=0, descending=True)
z = torch.clamp(x, min=0)  # Any values less than 0, it changes to zero

x = torch.tensor([1, 0, 1, 1, 1], dtype = torch.bool)
z = torch.any(x)
print(z)
z = torch.all(x)
print(z)

