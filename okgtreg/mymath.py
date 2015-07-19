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

# def MatrixInverse(M):
#     """
#     Calculate the inverse for a numpy matrix.
#
#     :param M:
#         M: Numpy matrix.
#
#     :return:
#         Inverse of M
#     """
#     D, P = np.linalg.eigh(M)
#     D = D[::-1]
#     P = P[:, ::-1]
#     D_inv = np.matrix(np.diag(1. / D))
#     M_inv = P * D_inv
#     M_inv = M_inv * P.T
#     return M_inv

# def ApplyICDonGramMatrix(K, centerK=True):
#     """
#     Apply Incomplete Cholesky Decomposition on an uncentered Gram matrix.
#
#     A Gram matrix K (n*n) assumes the following approximation:
#
#         K \approx G * G^T
#         G = U*S*V
#
#     where G is a n*m matrix (m < n), which can be SVD decomposed. So U^T * U = I.
#
#     So by combining them, we have:
#
#         K \approx U * Lambda * U^T
#
#     where Lambda = S^2, which is the m leading eigen values of K.
#
#     Reference:
#         2002, Bach and Jordan, Kernel Independent Component Analysis, Journal of Machine Learning Research
#
#     :param K: numpy matrix, uncentered Gram matrix
#
#     :param centerG: boolean, True by default.
#         If True, G is centered. In particular, centered G will be used to approximate the centered K, i.e.
#
#             (I - Ones/n) * K * (I - Ones/n) \approx G * G^T
#
#         after permuting K according to pind.
#
#     :return:
#         U: numpy matrix, a n*m lower triangular matrix such that U^T * U = I.
#         Lambda: 1-d numpy array, vector of m leading eigen-values of K.
#         pind: 1-d numpy array of int32, vector of permutation indices, which are the column numbers of K in the same order as ICD retains them.
#     """
#     # Center the Gram matrix first then apply decomposition
#     if centerK:
#         n = K.shape[0]
#         I = np.matrix(np.identity(n))
#         Ones = np.matrix(np.ones((n,n)))
#         # K = np.einsum('ij, jk, lk', I - Ones/n, K, I - Ones/n) # slow
#         K = (I - Ones/n) * K * (I - Ones/n)
#         K = (K + K.T)/2 # force symmetry
#
#     # ICD
#     reltol = 1e-6
#     L, m, pind = pyichol.ichol_full(K, reltol)
#     G = np.matrix(L[:m].T)
#
#     # if centerG:
#     #     n = G.shape[0]
#     #     I = np.identity(n)
#     #     Ones = np.matrix(np.ones((n, n)))
#     #     G = (I - Ones/n) * G  # centered G
#
#     # SVD
#     U, s, V = np.linalg.svd(G, full_matrices=False)
#     Lambda = s**2
#
#     return U, Lambda, pind

def ApplyICDonSymmetricMatrix(K, center=True):
    """
    Apply Incomplete Cholesky Decomposition on Gram matrix.

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

    :param center: boolean, True by default.
        If True, K = G * G^T is a centered matrix, i.e. for a un-centered Gram matrix K, we have

            (I - Ones/n) * K * (I - Ones/n) \approx G * G^T

        after permuting K according to pind.

    :return:
        U: numpy matrix. A n*m lower triangular matrix such that U^T * U = I.
        Lambda: 1-d numpy array. A vector of m leading eigen-values of K.
        pind: 1-d numpy array of int32. A vector of permutation indices, which are the column numbers of K in the same order as ICD retains them.
    """
    # Center the Gram matrix first then apply decomposition
    # if centerK:
    #     n = K.shape[0]
    #     I = np.matrix(np.identity(n))
    #     Ones = np.matrix(np.ones((n,n)))
    #     # K = np.einsum('ij, jk, lk', I - Ones/n, K, I - Ones/n) # slow
    #     K = (I - Ones/n) * K * (I - Ones/n)
    #     K = (K + K.T)/2 # force symmetry

    # ICD
    reltol = 1e-6 # fixed threshold
    L, m, pind = pyichol.ichol_full(K, reltol)
    G = np.matrix(L[:m].T)
    n = G.shape[0]

    if center:
        I = np.identity(n)
        Ones = np.matrix(np.ones((n, n)))
        G = (I - Ones/n) * G  # (column) centered G

    # SVD
    U, d, V = np.linalg.svd(G, full_matrices=False)
    Lambda = d**2 # eigen values of Gram matrix

    return U, Lambda, pind