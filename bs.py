import math


import numpy as np


def back_substitution(A, b):
    n = b.shape[0]
    x = np.zeros_like(b)
    for k in range(n - 1, -1, -1):
        x[k] = (b[k] - np.dot(A[k, k + 1:n], x[k + 1:n])) / A[k, k]

    return x


if __name__ == '__main__':
    A = np.array([[1, 0, 0, -3, 6, 0, 0],
                  [0, 1, 0, 2, 0, 0, 0],
                  [0, 0, 1, 0, 2, 0, 0],
                  [0, 0, 0, 1, 0, 0, -math.cos(12)],
                  [0, 0, 0, 0, 1, math.sin(12), 0],
                  [0, 0, 0, 0, 0, 1, 6],
                  [0, 0, 0, 0, 0, 0, 1]])
    b = np.array([0.0, 0, 0, 0, 0, 0, 1])
    x = back_substitution(A, b)
    print(x)