#This section of the chapter is named as "The Model" in the book

import Vectors as vec

#beta = [alpha, beta_1, ..., beta_k]

#x_i = [1, x_i1, ..., x_ik]

def predict(x_i, beta):
    """assumes that the first element of each x_i is 1"""
    return vec.dot(x_i, beta)


# x is a list of vectors which look like this
#[1, # constant term
#49, # number of friends
#4, # work hours per day
#0] # doesn't have PhD