# Imports
import numpy as np
import torch


# We can define a model as follows
def model(x):
    return x @ w.t() + b


# Defining a cost function
def mse(t1, t2):
    error = t1 - t2
    return torch.sum(error * error)/error.numel()


# Input (temp, rainfall, humidity)
input_array = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')


# Targets (apples, oranges)
target_array = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(input_array)
targets = torch.from_numpy(target_array)

w = torch.randn(2, 3, requires_grad=True)  # A 2x3 matrix
b = torch.randn(2, requires_grad=True)  # a Vector

preds = model(inputs)
print(preds)  # This is our model prediction, which is trash
print(targets)  # There is a huge difference between predicted model and actual values

loss = mse(preds, targets)
print("Before doing gradient descent: ", loss)
# To compute gradients
loss.backward()
# Gradients for weights and biases
w.grad
b.grad

with torch.no_grad():  # We use no_grad(), because we don't want to track
    w -= w.grad * 1e-4
    b -= b.grad * 1e-4

preds = model(inputs)
loss = mse(preds, targets)
print("Loss after gradient descent: ", loss) # We can see a decrease in the loss


# Resetting the gradients to zero
w.grad.zero_()
b.grad.zero_()
# Everytime we are done dealing with a gradient,
# Make it zero, otherwise pytorch accumulates the gradients on top of it
print(w.grad)
print(b.grad)


# Now let's do the same thing step by step but for many epochs
for i in range(1000):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()  # Computing the gradients
    with torch.no_grad():  # This halts the tracking of the gradient
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print("Loss after 1000 iterations of G.D: ", loss)

print(preds)  # Now the predictions are really good.
print(targets)