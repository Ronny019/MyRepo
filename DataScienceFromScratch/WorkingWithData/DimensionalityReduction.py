import Matrices as mat
import Rescaling as res

print('-----------------------------------------------')
def de_mean_matrix(A):
    """returns the result of subtracting from every value in A the mean
    value of its column. the resulting matrix has mean 0 in every column"""
    nr, nc = mat.shape(A)
    column_means, _ = scale(A)
    return mat.make_matrix(nr, nc, lambda i, j: A[i][j] - column_means[j])