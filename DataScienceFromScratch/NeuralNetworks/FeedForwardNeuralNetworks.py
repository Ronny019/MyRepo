import Vectors as vec
import math

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(vec.dot(weights, inputs))


def feed_forward(neural_network, input_vector):
    """takes in a neural network
    (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""
    outputs = []
    # process one layer at a time
    for layer in neural_network:
        input_with_bias = input_vector + [1] # add a bias input
        output = [neuron_output(neuron, input_with_bias) # compute the output
                 for neuron in layer] # for each neuron
        outputs.append(output) # and remember it
        # then the input to the next layer is the output of this one
        input_vector = output
    return outputs


xor_network = [# hidden layer
[[20, 20, -30], # 'and' neuron
[20, 20, -10]], # 'or' neuron
# output layer
[[-60, 60, -30]]] # '2nd input but not 1st input' neuron
for x in [0, 1]:
    for y in [0, 1]:
        # feed_forward produces the outputs of every neuron
        # feed_forward[-1] is the outputs of the output-layer neurons
        feed = feed_forward(xor_network,[x, y])
        print(x, y, feed[-1])