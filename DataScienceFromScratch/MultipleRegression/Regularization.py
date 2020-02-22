import Vectors as vec
import FittingTheModel as fit
from functools import partial
import random
import StochasticGradientDescent as sto
import GoodnessOfFit as good
# alpha is a *hyperparameter* controlling how harsh the penalty is
# sometimes it's called "lambda" but that already means something in Python

def ridge_penalty(beta, alpha):
    return alpha * vec.dot(beta[1:], beta[1:])

def squared_error_ridge(x_i, y_i, beta, alpha):
    """estimate error plus ridge penalty on beta"""
    return fit.error(x_i, y_i, beta) ** 2 + ridge_penalty(beta, alpha)

def ridge_penalty_gradient(beta, alpha):
    """gradient of just the ridge penalty"""
    return [0] + [2 * alpha * beta_j for beta_j in beta[1:]]

def squared_error_ridge_gradient(x_i, y_i, beta, alpha):
    """the gradient corresponding to the ith squared error term
    including the ridge penalty"""
    return vec.vector_add(fit.squared_error_gradient(x_i, y_i, beta),
                      ridge_penalty_gradient(beta, alpha))
def estimate_beta_ridge(x, y, alpha):
    """use gradient descent to fit a ridge 
    regression with penalty alpha"""
    beta_initial = [random.random() for x_i in x[0]]
    return sto.minimize_stochastic(partial(squared_error_ridge, alpha=alpha),
                                partial(squared_error_ridge_gradient,
                                alpha=alpha),
                                x, y,
                                beta_initial,
                                0.001)

print("----------------------------------------------------------------------")
random.seed(0)
beta_0 = estimate_beta_ridge(fit.x, fit.daily_minutes_good, alpha=0.0)
# [30.6, 0.97, -1.87, 0.91]
print(beta_0)
print(vec.dot(beta_0[1:], beta_0[1:])) # 5.26
print(good.multiple_r_squared(fit.x, fit.daily_minutes_good, beta_0)) # 0.680


beta_0_01 = estimate_beta_ridge(fit.x, fit.daily_minutes_good, alpha=0.01)
# [30.6, 0.97, -1.86, 0.89]
print(beta_0_01)
print(vec.dot(beta_0_01[1:], beta_0_01[1:])) # 5.26
print(good.multiple_r_squared(fit.x, fit.daily_minutes_good, beta_0_01)) # 0.680


beta_0_1 = estimate_beta_ridge(fit.x, fit.daily_minutes_good, alpha=0.1)
print(beta_0_1)
print(vec.dot(beta_0_1[1:], beta_0_1[1:])) # 5.26
print(good.multiple_r_squared(fit.x, fit.daily_minutes_good, beta_0_1)) # 0.680



beta_1 = estimate_beta_ridge(fit.x, fit.daily_minutes_good, alpha=1)
print(beta_1)
print(vec.dot(beta_1[1:], beta_1[1:])) # 5.26
print(good.multiple_r_squared(fit.x, fit.daily_minutes_good, beta_1)) # 0.680


beta_10 = estimate_beta_ridge(fit.x, fit.daily_minutes_good, alpha=10)
print(beta_10)
print(vec.dot(beta_10[1:], beta_10[1:])) # 5.26
print(good.multiple_r_squared(fit.x, fit.daily_minutes_good, beta_10)) # 0.680

def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])