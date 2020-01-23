import matplotlib.pyplot as plt
from functools import partial
def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

def square(x):
    return x * x

def derivative(x):
    return 2 * x

derivative_estimate = partial(difference_quotient, square, h=0.00001)

# plot to show they're basically the same

x = range(-10,10)
y= map(derivative,x)
plt.title("Actual Derivatives vs. Estimates")
plt.plot(x, list(map(derivative, x)), 'rx', label='Actual') # red x
plt.plot(x, list(map(derivative_estimate, x)), 'b+', label='Estimate') # blue +
plt.legend(loc=9)
plt.show()

def partial_difference_quotient(f, v, i, h):
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0) # add h to just the ith element of v
    for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
    for i, _ in enumerate(v)]