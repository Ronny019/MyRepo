import TheModel as mod
import Correlation as cor
import StochasticGradientDescent as sto

import random

def squared_error(x_i, y_i, theta):
    alpha, beta = theta
    return mod.error(alpha, beta, x_i, y_i) ** 2

def squared_error_gradient(x_i, y_i, theta):
    alpha, beta = theta
    return [-2 * mod.error(alpha, beta, x_i, y_i), # alpha partial derivative
            -2 * mod.error(alpha, beta, x_i, y_i) * x_i] # beta partial derivative
# choose random value to start
random.seed(0)
theta = [random.random(), random.random()]
alpha, beta = sto.minimize_stochastic(squared_error,
                                      squared_error_gradient,
                                      cor.num_friends_good,
                                      cor.daily_minutes_good,
                                      theta,
                                      0.0001)

print("----------------------------------------------------------")
print(alpha, beta)