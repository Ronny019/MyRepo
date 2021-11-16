import torch

x = torch.tensor([1,2,3,4,5], dtype=torch.float32)
y = torch.tensor([5,9,13,17,21], dtype=torch.float32)

LEARNING_RATE = 0.01
# This is the parameter we want to optimize -> requires_grad=True
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
for i in range(20):
    print(f"Epoch: {i}")
    y_predicted = (a * x) + b
    loss = ((y_predicted - y) ** 2).mean()
    print(loss)

    loss.backward()
    print(a.grad)
    print(b.grad)
    with torch.no_grad():
        a -= LEARNING_RATE * a.grad
        b -= LEARNING_RATE * b.grad
    # don't forget to zero the gradients
    a.grad.zero_()
    b.grad.zero_()
