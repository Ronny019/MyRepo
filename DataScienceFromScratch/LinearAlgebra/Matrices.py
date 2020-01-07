A = [[1, 2, 3], # A has 2 rows and 3 columns
[4, 5, 6]]
B = [[1, 2], # B has 3 rows and 2 columns
[3, 4],
[5, 6]]

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of elements in first row
    return num_rows, num_cols

print(shape(A))
print(shape(B))

def get_row(A, i):
    return A[i] # A[i] is already the ith row
def get_column(A, j):
    return [A_i[j] # jth element of row A_i
    for A_i in A]

print(get_row(A,1))
print(get_column(B,0))