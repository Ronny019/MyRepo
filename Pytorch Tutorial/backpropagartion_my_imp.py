import torch

x = torch.tensor(5.0)
y = torch.tensor(25.0)

LEARNING_RATE = 0.01
# This is the parameter we want to optimize -> requires_grad=True
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    print(f"Epoch: {i}")
    y_predicted = (a * x) + b
    loss = (y_predicted - y) ** 2
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
