import torch

x = torch.arange(9)
x_3x3 = x.view(3, 3)
print(x, "\n", x_3x3)
x_3x3 = x.reshape(3, 3)

x1 = torch.rand((2, 5)) * 20
x2 = torch.rand((2, 5))

print(torch.cat((x1, x2), dim=0))

z = x1.view(-1) # Flatten the array
print(z)

batch = 64
x = torch.rand((batch, 2, 5))
print(x.shape)
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)  # Special case of transpose
print(z.shape)

x = torch.arange(10)
print(x.shape)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
z = x.squeeze(1)
print(x.shape)
print(z.shape)