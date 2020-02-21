import FittingTheModel as fit

def multiple_r_squared(x, y, beta):
    sum_of_squared_errors = sum(fit.error(x_i, y_i, beta) ** 2
                                for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)


