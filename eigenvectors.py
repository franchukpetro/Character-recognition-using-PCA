import numpy as np
from numpy import linalg


# Find eigenvalues and eigenvectors using numpy
def eiginevectors(S):
    S = np.array(S)
    ev, evc = linalg.eig(S)

    # sorting eigenvalues from largest to smallest (eigenvectors are sorted respectively to eigenvalues)
    idx = ev.argsort()[::-1]
    evc = evc[:, idx]

    evectors = [[lst[i] for lst in evc] for i in range(len(evc))]
    return evectors


# Maps evc to bigger dimension
def map_back(matrix, evcs):
    mapped_evc = []
    for evc in evcs:
        mapped_evc.append(np.dot(matrix, evc))
    return mapped_evc
