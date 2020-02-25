import Vectors as vec

def step_function(x):
    return 1 if x >= 0 else 0


def perceptron_output(weights, bias, x):
    """returns 1 if the perceptron 'fires', 0 if not"""
    calculation = vec.dot(weights, x) + bias
    return step_function(calculation)

# AND Gate
weights = [2, 2]
bias = -3

# OR Gate
weights = [2, 2]
bias = -1

# NOT Gate
weights = [-2]
bias = 1


and_gate = min
or_gate = max
xor_gate = lambda x, y: 0 if x == y else 1