import random
import OverfittingAndUnderfitting as ovund
import functools
import TheLogisticFunction as logfn
import ChoosingTheRightStep as choose
import TheProblem as prob
import StochasticGradientDescent as sto

random.seed(0)
x_train, x_test, y_train, y_test = ovund.train_test_split(prob.rescaled_x, prob.y, 0.33)
# want to maximize log likelihood on the training data
fn = functools.partial(logfn.logistic_log_likelihood, x_train, y_train)
gradient_fn = functools.partial(logfn.logistic_log_gradient, x_train, y_train)
# pick a random starting point
beta_0 = [random.random() for _ in range(3)]
# and maximize using gradient descent
beta_hat = choose.maximize_batch(fn, gradient_fn, beta_0)

print("-----------------------------------------------------------------------")
print(beta_hat)

beta_hat2 = sto.maximize_stochastic(logfn.logistic_log_likelihood_i,
                                logfn.logistic_log_gradient_i,
                                x_train, y_train, beta_0)

print(beta_hat2)

#x_train_unscaled, x_test_unscaled, y_train_unscaled, y_test_unscaled = ovund.train_test_split(prob.x, prob.y, 0.33)

#fn_unscaled = functools.partial(logfn.logistic_log_likelihood, x_train_unscaled, y_train_unscaled)
#gradient_fn_unscaled = functools.partial(logfn.logistic_log_gradient, x_train_unscaled, y_train_unscaled)

#beta_hat_unscaled = choose.maximize_batch(fn_unscaled, gradient_fn_unscaled, beta_0)

#print(beta_hat_unscaled)