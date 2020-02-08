import Matrices as mat
import Rescaling as res
import Vectors as vec
from functools import partial
import ChoosingTheRightStep as choose
import StochasticGradientDescent as sto

print('-----------------------------------------------')
def de_mean_matrix(A):
    """returns the result of subtracting from every value in A the mean
    value of its column. the resulting matrix has mean 0 in every column"""
    nr, nc = mat.shape(A)
    column_means, _ = res.scale(A)
    return mat.make_matrix(nr, nc, lambda i, j: A[i][j] - column_means[j])

def direction(w):
    mag = vec.magnitude(w)
    return [w_i / mag for w_i in w]

def directional_variance_i(x_i, w):
    """the variance of the row x_i in the direction determined by w"""
    return vec.dot(x_i, direction(w)) ** 2

def directional_variance(X, w):
    """the variance of the data in the direction determined w"""
    return sum(directional_variance_i(x_i, w)
               for x_i in X)


def directional_variance_gradient_i(x_i, w):
    """the contribution of row x_i to the gradient of
    the direction-w variance"""
    projection_length = vec.dot(x_i, direction(w))
    return [2 * projection_length * x_ij for x_ij in x_i]

def directional_variance_gradient(X, w):
    return vec.vector_sum(directional_variance_gradient_i(x_i,w)
                      for x_i in X)

def first_principal_component(X):
    guess = [1 for _ in X[0]]
    unscaled_maximizer = choose.maximize_batch(
            partial(directional_variance, X), # is now a function of w
            partial(directional_variance_gradient, X), # is now a function of w
            guess)
    return direction(unscaled_maximizer)


# here there is no "y", so we just pass in a vector of Nones
# and functions that ignore that input
def first_principal_component_sgd(X):
    guess = [1 for _ in X[0]]
    unscaled_maximizer = sto.maximize_stochastic(
                        lambda x, _, w: directional_variance_i(x, w),
                        lambda x, _, w: directional_variance_gradient_i(x, w),
                        X,
                        [None for _ in X], # the fake "y"
                        guess)
    return direction(unscaled_maximizer)