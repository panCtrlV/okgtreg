__author__ = 'panc'

import numpy as np
import pymc.gp.incomplete_chol as pyichol

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

def ApplyICDonGramMatrix(K, centerG=True):
    """
    Apply Incomplete Cholesky Decomposition on an uncentered Gram matrix.

    A Gram matrix K (n*n) assumes the following approximation:

        K \approx G * G^T
        G = U*S*V

    where G is a n*m matrix (m < n), which can be SVD decomposed. So U^T * U = I.

    So by combining them, we have:

        K \approx U * Lambda * U^T

    where Lambda = S^2, which is the m leading eigen values of K.

    Reference:
        2002, Bach and Jordan, Kernel Independent Component Analysis, Journal of Machine Learning Research

    :param K: numpy matrix, uncentered Gram matrix

    :param centerG: boolean, True by default.
        If True, G is centered. In particular, centered G will be used to approximate the centered K, i.e.

            (I - Ones/n) * K * (I - Ones/n) \approx G * G^T

        after permuting K according to pind.

    :return:
        U: numpy matrix, a n*m lower triangular matrix such that U^T * U = I.
        Lambda: 1-d numpy array, vector of m leading eigen-values of K.
        pind: 1-d numpy array of int32, vector of permutation indices, which are the column numbers of K in the same order as ICD retains them.
    """
    reltol = 1e-6
    L, m, pind = pyichol.ichol_full(K, reltol)
    G = np.matrix(L[:m].T)

    if centerG:
        n = G.shape[0]
        I = np.identity(n)
        Ones = np.matrix(np.ones((n, n)))
        G = (I - Ones/n) * G  # centered G

    U, s, V = np.linalg.svd(G, full_matrices=False)
    Lambda = s**2
    return U, Lambda, pind
