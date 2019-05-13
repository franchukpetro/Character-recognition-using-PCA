import numpy as np


# Transforms matrix into mean deviation form
def to_mean_deviation_form(matrix, mean):
    n = len(matrix)
    m = len(matrix[0])

    for row in range(n):
        for col in range(m):
            matrix[row][col] -= mean[row]
    return matrix


# Find covariance matrix
def cov_matrix(B):
    n = len(B[0])
    m = np.matmul(B.transpose(), B)
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] *= 1 / (n - 1)
    return m
