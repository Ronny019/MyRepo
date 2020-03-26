import Matrices as mat
import Vectors as vec

def matrix_product_entry(A, B, i, j):
    return vec.dot(mat.get_row(A, i), mat.get_column(B, j))

def matrix_multiply(A, B):
    n1, k1 = mat.shape(A)
    n2, k2 = mat.shape(B)
    if k1 != n2:
        raise ArithmeticError("incompatible shapes!")
    return mat.make_matrix(n1, k2, partial(matrix_product_entry, A, B))

#v = [1, 2, 3]
#v_as_matrix = [[1],
#              [2],
#              [3]]

def vector_as_matrix(v):
    """returns the vector v (represented as a list) as a n x 1 matrix"""
    return [[v_i] for v_i in v]

def vector_from_matrix(v_as_matrix):
    """returns the n x 1 matrix as a list of values"""
    return [row[0] for row in v_as_matrix]


def matrix_operate(A, v):
    v_as_matrix = vector_as_matrix(v)
    product = matrix_multiply(A, v_as_matrix)
    return vector_from_matrix(product)

def find_eigenvector(A, tolerance=0.00001):
    guess = [random.random() for __ in A]
    while True:
        result = matrix_operate(A, guess)
        length = magnitude(result)
        next_guess = scalar_multiply(1/length, result)
        if distance(guess, next_guess) < tolerance:
            return next_guess, length # eigenvector, eigenvalue
        guess = next_guess