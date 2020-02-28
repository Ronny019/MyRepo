import random
import Backpropagation as back
import FeedForwardNeuralNetworks as feed
import matplotlib
import matplotlib.pyplot as plt
raw_digits = [
          """11111
             1...1
             1...1
             1...1
             11111""",

          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",

          """11111
             ....1
             11111
             1....
             11111""",

          """11111
             ....1
             11111
             ....1
             11111""",

          """1...1
             1...1
             11111
             ....1
             ....1""",

          """11111
             1....
             11111
             ....1
             11111""",

          """11111
             1....
             11111
             1...1
             11111""",

          """11111
             ....1
             ....1
             ....1
             ....1""",

          """11111
             1...1
             11111
             1...1
             11111""",

          """11111
             1...1
             11111
             ....1
             11111"""]

def make_digit(raw_digit):
    return [1 if c == '1' else 0
            for row in raw_digit.split("\n")
            for c in row.strip()]

inputs = list(map(make_digit, raw_digits))

targets = [[1 if i == j else 0 for i in range(10)]
            for j in range(10)]

random.seed(0) # to get repeatable results
input_size = 25 # each input is a vector of length 25
num_hidden = 5 # we'll have 5 neurons in the hidden layer
output_size = 10 # we need 10 outputs for each input
# each hidden neuron has one weight per input, plus a bias weight
hidden_layer = [[random.random() for __ in range(input_size + 1)]
                for __ in range(num_hidden)]
# each output neuron has one weight per hidden neuron, plus a bias weight
output_layer = [[random.random() for __ in range(num_hidden + 1)]
                for __ in range(output_size)]

# the network starts out with random weights
network = [hidden_layer, output_layer]

for i in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        back.backpropagate(network, input_vector, target_vector)
    print("Training round ",i," complete")

def predict(input):
    return feed.feed_forward(network, input)[-1]


print("------------------------------------------------------------")
print(predict(inputs[7]))
# [0.026, 0.0, 0.0, 0.018, 0.001, 0.0, 0.0, 0.967, 0.0, 0.0]

three_pred = predict([0,1,1,1,0, # .@@@.
                      0,0,0,1,1, # ...@@
                      0,0,1,1,0, # ..@@.
                      0,0,0,1,1, # ...@@
                      0,1,1,1,0]) # .@@@.
                      # [0.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.01, 0.0, 0.12]

print(three_pred)

pred_eight = predict([0,1,1,1,0, # .@@@.
                    1,0,0,1,1, # @..@@
                    0,1,1,1,0, # .@@@.
                    1,0,0,1,1, # @..@@
                    0,1,1,1,0]) # .@@@.
                    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.0, 0.0, 0.93, 1.0]

print(pred_eight)


weights = network[0][0] # first neuron in hidden layer
abs_weights = list(map(abs, weights)) # darkness only depends on absolute value
grid = [abs_weights[row:(row+5)] # turn the weights into a 5x5 grid
        for row in range(0,25,5)] # [weights[0:5], ..., weights[20:25]]
ax = plt.gca() # to use hatching, we'll need the axis
ax.imshow(grid, # here same as plt.imshow
         cmap=matplotlib.cm.binary, # use white-black color scale
         interpolation='none') # plot blocks as blocks


def patch(x, y, hatch, color):
    """return a matplotlib 'patch' object with the specified
    location, crosshatch pattern, and color"""
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
    hatch=hatch, fill=False, color=color)


# cross-hatch the negative weights
for i in range(5): # row
    for j in range(5): # column
        if weights[5*i + j] < 0: # row i, column j = weights[5*i + j]
            # add black and white hatches, so visible whether dark or light
            ax.add_patch(patch(j, i, '/', "white"))
            ax.add_patch(patch(j, i, '\\', "black"))

plt.show()


left_column_only = [1, 0, 0, 0, 0] * 5
print (feed.feed_forward(network, left_column_only)[0][0]) # 1.0
center_middle_row = [0, 0, 0, 0, 0] * 2 + [0, 1, 1, 1, 0] + [0, 0, 0, 0, 0] * 2
print (feed.feed_forward(network, center_middle_row)[0][0]) # 0.95
right_column_only = [0, 0, 0, 0, 1] * 5
print (feed.feed_forward(network, right_column_only)[0][0]) # 0.0


my_three = [0,1,1,1,0, # .@@@.
            0,0,0,1,1, # ...@@
            0,0,1,1,0, # ..@@.
            0,0,0,1,1, # ...@@
            0,1,1,1,0] # .@@@.
hidden, output = feed.feed_forward(network, my_three)

print(hidden)

print(output)