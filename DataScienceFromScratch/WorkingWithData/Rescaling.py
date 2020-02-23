import Vectors as vec
import DescribingASingleSetOfData as desc
import Matrices as mat
print('------------------------------------------------------------------')
a_to_b = vec.distance([63, 150], [67, 160]) # 10.77
a_to_c = vec.distance([63, 150], [70, 171]) # 22.14
b_to_c = vec.distance([67, 160], [70, 171]) # 11.40

print(a_to_b,a_to_c,b_to_c)

a_to_b = vec.distance([160, 150], [170.2, 160]) # 14.28
a_to_c = vec.distance([160, 150], [177.8, 171]) # 27.53
b_to_c = vec.distance([170.2, 160], [177.8, 171]) # 13.37

print(a_to_b,a_to_c,b_to_c)


def scale(data_matrix):
    """returns the means and standard deviations of each column"""
    num_rows, num_cols = mat.shape(data_matrix)
    means = [desc.mean(mat.get_column(data_matrix,j))
             for j in range(num_cols)]
    stdevs = [desc.standard_deviation(mat.get_column(data_matrix,j))
             for j in range(num_cols)]
    return means, stdevs


def rescale(data_matrix):
    """rescales the input data so that each column
    has mean 0 and standard deviation 1
    leaves alone columns with no deviation"""
    means, stdevs = scale(data_matrix)
    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]
    num_rows, num_cols = mat.shape(data_matrix)
    return mat.make_matrix(num_rows, num_cols, rescaled)