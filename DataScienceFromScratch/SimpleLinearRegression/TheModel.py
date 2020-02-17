import Correlation as cor
import DescribingASingleSetOfData as desc
def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    """the error from predicting beta * x_i + alpha
    when the actual value is y_i"""
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
                for x_i, y_i in zip(x, y))



def least_squares_fit(x, y):
    """given training values for x and y,
    find the least-squares values of alpha and beta"""
    beta = cor.correlation(x, y) * desc.standard_deviation(y) / desc.standard_deviation(x)
    alpha = desc.mean(y) - beta * desc.mean(x)
    return alpha, beta

alpha, beta = least_squares_fit(cor.num_friends_good, cor.daily_minutes_good)

print("-------------------------------------------------------------------------")
print("Alpha ",alpha," Beta ",beta)

def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in desc.de_mean(y))


def r_squared(alpha, beta, x, y):
    """the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model"""
    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) /
                  total_sum_of_squares(y))

r_sq = r_squared(alpha, beta, cor.num_friends_good, cor.daily_minutes_good) # 0.329

print(r_sq)