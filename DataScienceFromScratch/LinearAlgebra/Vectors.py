import functools as ft
def vector_add(v, w):
    """adds corresponding elements"""
    return [v_i + w_i
    for v_i, w_i in zip(v, w)]

def vector_subtract(v, w):
    """subtracts corresponding elements"""
    return [v_i - w_i
    for v_i, w_i in zip(v, w)]

def vector_sum(vectors):
    """sums all corresponding elements"""
    result = vectors[0] # start with the first vector
    for vector in vectors[1:]: # then loop over the others
        result = vector_add(result, vector) # and add them to the result
    return result

def vector_sum1(vectors):
    return ft.reduce(vector_add, vectors)

lis1 = [ 1 , 3, 5, 6, 2 ]
lis2 = [ 0 , 2, 4, 5, 1 ]

vectorlist = [[ 1 , 3, 5, 6, 2 ],[ 0 , 2, 4, 5, 1 ]]

print(vector_add(lis1,lis2));

print(vector_subtract(lis1,lis2));

print(vector_sum(vectorlist));

print(vector_sum1(vectorlist));

def scalar_multiply(c, v):
    """c is a number, v is a vector"""
    return [c * v_i for v_i in v]

print(scalar_multiply(3,lis1))

def vector_mean(vectors):
    """compute the vector whose ith element is the mean of the
    ith elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

print(vector_mean(vectorlist))

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i
    for v_i, w_i in zip(v, w))

print(dot(lis1,lis2))