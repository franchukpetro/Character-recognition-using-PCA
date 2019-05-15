import numpy as np
from numpy import linalg


# Find eigenvalues and eigenvectors using numpy
def eiginevectors(S):
    ev, evc = linalg.eig(S)
    # sorting eigenvalues from largest to smallest (eigenvectors are sorted respectively to eigenvalues)
    return evc[:, ev.argsort()[::-1]]


# Maps evc to bigger dimension
def map_back(matrix, evcs):
    return np.matmul(matrix, np.array(evcs).T).T
