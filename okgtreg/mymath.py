__author__ = 'panc'

import numpy as np

def PSDMatrixSqrt(x):
    '''
    Compute the square root for a PSD matrix, using eigen decomposition.

    **Input**
        X: numpy matrix, a PSD matrix.

    **Output**
        Xsqrt: numpy matrix, the square root of <X>.
    '''
    if type(x).__moduel__ != "numpy.matrixlib.defmatrix":
        raise Exception('[Error] The input x is not a Numpy matrix!')

    D, P = np.linalg.eig(x)
    Dsqrt = np.sqrt(D)
    Dsqrt = np.sqrt(D)
    Dsqrt[Dsqrt < 0.] = 0.
    return P.T * np.diag(Dsqrt) * P

def MatrixInverse(M):
    """
    Calculate the inverse for a numpy matrix.

    :param M:
        M: Numpy matrix.

    :return:
        Inverse of M
    """
    D, P = np.linalg.eigh(M)
    D = D[::-1]
    P = P[:, ::-1]
    D_inv = np.matrix(np.diag(1. / D))
    M_inv = P * D_inv
    M_inv = M_inv * P.T
    return M_inv

