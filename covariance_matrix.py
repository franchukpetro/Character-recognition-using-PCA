import numpy as np


# Transforms matrix into mean deviation form
def to_mean_deviation_form(matrix, mean):
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            matrix[row][col] -= mean[row]
    return matrix

# Find covariance matrix
def cov_matrix(B):
    return np.matmul(B.T, B) / (B.shape[1] - 1)
