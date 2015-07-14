__author__ = 'panc'


"""
    Functions for grouped optimal transformation using kernel method.
    Implemented as a class
"""

import numpy as np
import scipy as sp
import scipy.linalg as slinalg
from sklearn.metrics.pairwise import pairwise_distances

from mymath import *
from kernel_selector import *


class OKGTReg:
    def __init__(self, x, y, xKernelNames, yKernelName, xKernelParams, yKernelParam, eps=1e-5, **kwargs):
        """
        ** Input: **
            X: 	np.matrix.
                Design matrix.

            y: 	np.matrix.
                Response observations.

            xKernelNames:	list of strings.
                            Kernel names for each group of the predictor variables.
                            Its size should be the same as the size of <xGroup>.

                            Current implemented:
                                Gaussian,
                                Laplace

            yKernelName:	list of one string.
                            Though <y> is uni-variate, the kernel name is still passed a list, for example ["Laplace"]

            xKernelParas:	list of dictionary / dictionaries.
                            Each element in the list is a dictionary. A dictionary contains the name
                            of the parameters and the values.
                            For example:
                                if (X_1, X_2), (X_3, X_4), (X_5, X_6, X_7) is the grouping
                                and (Gaussian, Laplace, Gaussian) are the kernels, then the
                                parameter list can be
                                    [{sigma:0.5}, {alpha:0.2}, {sigma:0.1}].

            yKernelPara:

            kwargs:	provides <xGroup> if <X> is of more than one dimensional. It is a list of lists.
                    Grouping information of the predictors.
                    For example:
                        If the grouping of 7 predictor variables is as
                            (X_1, X_2), (X_3, X_4), (X_5, X_6, X_7),
                        then the value of <xGroup> should be:
                            [[1,2], [3,4], [5,6,7]].
                    The advantage of this format is that it can handle un-ordered grouping.
                    For example:
                        xGroup = [[1,3], [2,5], [4,6,7]]
                        indicates a different grouping from the previous example.
        """
        self.x = x
        self.y = y
        self.eps = eps

        p = x.shape[1]

        if p>1:
            if len(kwargs) == 0:
                self.xGroup = [[i+1] for i in range(0,p)]
                print '[Warning] Group partition is missing. Default partition (fully additive) is used.'
            else:
                self.xGroup = kwargs['xGroup'] # if provided, the argument name should be 'xGroup'
        else:
            self.xGroup = [[1]] # if p!=1, only one group

        if len(self.xGroup) == len(xKernelNames):
            self.xKernel_fns = OKGTReg.ConstructKernelFns(xKernelNames, xKernelParams)
            self.yKernel_fn = OKGTReg.ConstructKernelFns(yKernelName, yKernelParam)
        else:
            raise Exception("[Error] Number of groups and number of kernel functions don't match!")

        # TODO: Once instantiated, functions are estimated automatically
        # self.g, self.f, self.r2 = self.TrainOKGT(self.y, self.x, self.yKernel_fn, self.xKernel_fns, xGroup=self.xGroup)


    @staticmethod
    def ConstructKernelFns(names, params):
        """
        Given a list of kernel names and parameters, construct a list of kernel functions.

        ** Input **
            kernelNames: a string or a list of kernel name strings.
            kernelParas: a dictionary or a list of dictionaries.
                         For example:
                            [dict(sigma=0.1), dict(sigma=0.2), dict(sigma=0.3)]

        ** Output **
            LIST of kernel functions, each is a return of <KernelSelector> function.
            When you want to use the functions, use index to select the kernel function.
        """
        if isinstance(names, basestring):
            return [KernelSelector(names, **params)]
        else:
            l = len(names)

            if len(params) < l:
                raise Exception('[Error] Please provide parameters for each kernel function!')
            elif len(params) > l:
                raise Exception('[Error] Too many parameters!')

            return [ KernelSelector(names[i], **params[i]) for i in range(l) ]

    @staticmethod
    def GramMatrix(x, kernel_fn, centered=True):
    # def GramMatrix(x, kernel_fn):
        """
        Construct the Gram / centered Gram matrix from the data <x>. Each row of <x> is an observation.

        ** Input **
            x: 	numpy.matrix. (part of) the data matrix. If it is the reponse y, then it is a n*1 matrix (one column).
                If it is the predictor, the number of columns of the matrix is the same as the size of a group.
                For example,
                    if x = (X_1, X_2), then it is a n*2 matrix.

            kernel_fn: a bivariate kernel function, callable returne from <KernelSelector>

            centered: logical. Indicate if the gram matrix is centered.

        ** Output **
            GramMtx: numpy.matrix. Gram matrix.
        """
        n = len(x) # sample size
        # G = distance.pdist(x, metric=kernel_fn) # condensed distance matrix
        # G = distance.squareform(G) # convert to a square form
        # np.fill_diagonal(G, 1.) # fill the diagonal with 1's
        # G = np.matrix(G)
        G = np.matrix(pairwise_distances(x, metric=kernel_fn))
        # u,s,v = np.linalg.svd(G)
        # G = u * np.diag(np.abs(s)) * u.T

        if centered:
            I = np.identity(n)
            II = np.matrix(np.ones((n, n)))
            G = (I - II/n) * G * (I - II/n) # centered Gram matrix
            return (G + G.T)/2 # numerical issue cause asymmetry
        else:
            return G

    # -- GET ICD-REDUCED U AND LAMBDA FROM A GRAM MATRIX --- #
    def ReducedGramMatrix_ICD(self, x, kernel_fn):
        """
        Get U and Lambda from a Gram matrix.
        The Gram matrix is constructed from the given data and kernel function.

        :param x: numpy matrix.
                (part of) the data matrix. If it is the reponse y, then it is a n*1 matrix (one column).
                If it is the predictor, the number of columns of the matrix is the same as the size of a group.
                For example,
                    if x = (X_1, X_2), then it is a n*2 matrix.

        :param kernel_fn: callable returne from <KernelSelector>. It is a bivariate kernel function.

        :return:
            U: numpy matrix, a n*m lower triangular matrix such that U^T * U = I.
            Lambda: 1-d numpy array, vector of m leading eigen-values of K.
        """
        K = np.matrix(pairwise_distances(x, metric=kernel_fn))
        U, Lambda, pind = ApplyICDonGramMatrix(K)
        return U, Lambda

    # TODO: stack matrics
    def RowStackUandLambda(self, x, kernelFnsList, **kwargs):
        """
        Construct the row stack block matrix: [U_1, U_2, ..., U_l].

        :param x: numpy matrix. Design matrix for predictor variables.
        :param kernelFnsList: list of kernel functions, each is a callable returned from kernel_selector.KernelSelector.
        :param kwargs: additional keyword arguments.
                    If x is multivariate or high dimensional, provide the group information.
                    List of lists, same as <xGroup>.

        :return:
            numpy.matrix
            numpy.array
        """
        l = len(kernelFnsList)
        n,p = x.shape

        if p > 1:
            if len(kwargs) == 0:
                raise Exception("[Error] Data has more than one dimension, please provide the group structure 'xGroup'!")
            else:
                xGroup = kwargs['xGroup']
        else:
            xGroup = [[1]]

        if len(xGroup) != l:
            raise Exception("[Error] Number of groups is different from the number of kernels!")

        U_list = []
        Lambda_list = []
        for i in range(l):
            ind = [a-1 for a in xGroup[i]] # variable indices for group i
            x_groupi = x[:,ind]
            U_i, Lambda_i = self.ReducedGramMatrix_ICD(x_groupi, kernelFnsList[i])
            U_list.append(U_i)
            Lambda_list.append(Lambda_i)

        URowStack = np.hstack(U_list)
        LambdaStack = np.hstack(Lambda_list)

        return URowStack, LambdaStack


    @staticmethod
    def CovOperator_directSum(x, kernelFnsList, **kwargs):
        """
        Construct the covariance operator for a Hilbert space.
        For a single variables X, the estimate of its covariance operator is 'K_X*K_X', i.e. the product of its Gram matrix.

        If x is multivariate, a covariance operator matrix for the direct sum space is returned.
        Regulerization term, epsilon, is not added.

        Input:
            x: data matrix, can be multivariate
            kernelFnsList: a list of kernel functions, each of them is returned from <KernelSelector>.
                        It can be constructed by calling <ConstructKernelFns>.
            kwargs:	if x is multivariate or high dimensional, provide the group information.
                    List of lists, same as <xGroup>.

        Return:
            Rxx: covariance operator
            x_cGram_colBlock: stacked gram matrix
        """
        l = len(kernelFnsList)
        n, p = x.shape

        if p > 1:
            if len(kwargs) == 0:
                raise Exception("[Error] Data has more than one dimension, please provide the group structure 'xGroup'!")
            else:
                xGroup = kwargs['xGroup']
        else:
            xGroup = [[1]]

        if len(xGroup) != l:
            raise Exception("[Error] Number of groups is different from the number of kernels!")

        x_cG_colBlock = []
        for i in range(l):
            ind = [a-1 for a in xGroup[i]]
            x_groupi = x[:,ind]
            x_groupi_cG = OKGTReg.GramMatrix(x_groupi, kernelFnsList[i])
            x_cG_colBlock.append([x_groupi_cG])
        x_cG_colBlock = np.bmat(x_cG_colBlock)

        Rxx = x_cG_colBlock * x_cG_colBlock.T / n

        return (Rxx, x_cG_colBlock)

    @staticmethod
    def CrossCovOperator_directSum(y, x, yKernelFnList, xKernelFnsList, **kwargs):
        """
        Construct the cross-covariance operator.

        ** Input **
            y: response vector, must be univaraite
            x: design matrix, can be multivariate
            yKernelFnList: a list of a single kernel function, e.g. [fn()], it is returned from <KernelSelector>
            xKernelFnsList: a list of kernel functions, e.g. [fn1(), fn2(), ...], each of them is returned from <KernelSelector>

            **kwargs:	if x is multivariate or high dimensional, provide the group information.
                        List of lists, same as <xGroup>.
        ** Output **
            Ryx: cross-covariance operator Hx->Hy.
            y_cGram: centered gram matrix of y.
            x_cGram_colBlock: stacked centered gram matrices of x.
        """
        n, p = x.shape
        l = len(xKernelFnsList)

        if p>1:
            if len(kwargs) == 0:
                raise Exception("[Error] X has more than one dimension, please provide the group structure 'xGroup'!")
            else:
                xGroup = kwargs['xGroup']
                if len(xGroup) != l:
                    raise Exception("[Error] Number of groups in 'xGroup' is different from the number of kernel functions in 'xKernelFnList'!")
        else:
            xGroup = [[1]]

        x_cGram_colBlock = []
        for i in range(l):
            ind = [a-1 for a in xGroup[i]] # index of i-th group
            x_group_i = x[:,ind] # data of i-th group
            x_group_i_cGram = OKGTReg.GramMatrix(x_group_i, xKernelFnsList[i])
            x_cGram_colBlock.append([x_group_i_cGram])

        x_cGram_colBlock = np.bmat(x_cGram_colBlock)

        y_cGram = OKGTReg.GramMatrix(y, yKernelFnList[0])

        Ryx = y_cGram * x_cGram_colBlock.T / n

        return(Ryx, y_cGram, x_cGram_colBlock)


    # --- TRAIN OKGT --- #
    def TrainOKGT(self, y, x, yKernelFnList, xKernelFnsList, **kwargs):
        """
        Estimate the kernel optimal transformations.

        ** Input **
            y: numpy.matrix. A column vector.
            x: numpy.matrix. A predictor matrix.
            yKernel_fn: a list of a single kernel function for y.
            xKernel_fns: list of kernel functions for x.

            **kwargs: optional named arguments, including
                xGroup: list of lists. Group information for predictors. When x has >1 dimension, it must be provided.
        """
        print "== Start OKGT Training ==="
        eps = self.eps
        n, p = x.shape
        l = len(xKernelFnsList)

        if p > 1:
            if len(kwargs) == 0:
                raise Exception("[Error] X has more than one dimension, group structure <xGroup> is missing!")
            else:
                xGroup = kwargs['xGroup']
        else:
            xGroup = [[1]]

        if len(xGroup) != l:
            raise Exception("[Error] Number of groups in 'xGroup' is different from the number of kernel functions in 'xKernelFnList'!")

        Rxx, Gx = OKGTReg.CovOperator_directSum(x, xKernelFnsList, xGroup = xGroup)
        Ryy, Gy = OKGTReg.CovOperator_directSum(y, yKernelFnList)
        Ryx = Gy * Gx.T / n

        D, P = np.linalg.eigh(Ryy + eps * np.identity(n))
        D = D[::-1]
        P = P[:, ::-1]
        D_inv = np.matrix(np.diag(1. / np.sqrt(D)))
        Gy_inv = D_inv * P.T # Ryy^{-1/2}

        Rxx_inv = np.linalg.inv(Rxx + eps * np.identity(n*l))

        VyxVxy = Gy_inv * Ryx * Rxx_inv * Ryx.T * Gy_inv.T
        #TODO: if Rxx is large, the inverse would be slow.

        # g: optimal transformation for y
        r2, beta = slinalg.eigh(VyxVxy, eigvals=(n-1, n-1)) # only need the largest eigen value and vector
        beta = np.matrix(beta)
        zeta = D_inv * beta
        zeta = P * zeta
        g_opt = Gy * zeta

        # f: optimal transformation for x
        x_i = Ryx.T * g_opt
        x_i = Rxx_inv * x_i
        f_opt_ls = []
        for i in range(l):
            x_ii = x_i[i*n : (i+1)*n]
            Gx_i = Gx[i*n : (i+1)*n, :]
            f_i_opt = Gx_i * x_ii
            f_i_norm = np.sqrt(x_ii.T * f_i_opt)
            f_i_opt = f_i_opt / f_i_norm
            f_opt_ls.append(f_i_opt)

        f_opt = np.column_stack(f_opt_ls)

        return(g_opt, f_opt, float(r2))


    # --------------
    # OKGT traning using Incomplete Cholesky Decomposition
    # ---------------
    def TrainOKGT_ICD(self, y, x, yKernelFnList, xKernelFnsList, **kwargs):
        """

        :param y:
        :param x:
        :param yKernelFnList:
        :param xKernelFnsList:
        :param kwargs:
        :return:
        """
        print "== Start OKGT Training with ICD Invoked ==="
        eps = self.eps
        n, p = x.shape
        l = len(xKernelFnsList)

        if p > 1:
            if len(kwargs) == 0:
                raise Exception("[Error] X has more than one dimension, group structure <xGroup> is missing!")
            else:
                xGroup = kwargs['xGroup']
        else:
            xGroup = [[1]]

        if len(xGroup) != l:
            raise Exception("[Error] Number of groups in 'xGroup' is different from the number of kernel functions in 'xKernelFnList'!")

        U_y, Lambda_y = self.ReducedGramMatrix_ICD(y, yKernelFnList[0])
        U_x, Lambda_x = self.RowStackUandLambda(x, xKernelFnsList, xGroup = xGroup)

        R = np.matrix(np.diag(Lambda_x + eps)**2) + \
            np.matrix(np.diag(Lambda_x)) * (U_x.T * U_x - np.identity(U_x.shape[1])) * np.matrix(np.diag(Lambda_x))
        R_inv = np.linalg.inv(R)
        A = np.matrix(np.diag(Lambda_x / (Lambda_x + eps)))
        Uyx = U_y.T * U_x
        Lmd_y = np.matrix(np.diag(Lambda_y / (Lambda_y + eps)))

        # VV = U_y * Lmd_y * Uyx.T * A * Uyx * Lmd_y * U_y.T
        # print Lmd_y.shape, Uyx.shape, A.shape, R_inv.shape, A.shape, Uyx.T.shape, Lmd_y.shape, '\n'

        VV_reduced = Lmd_y * Uyx * A * R_inv * A * Uyx.T * Lmd_y
        m = VV_reduced.shape[0]

        r2, b = slinalg.eigh(VV_reduced, eigvals=(m-1, m-1))
        b = np.matrix(b)
        zeta = np.matrix(np.diag(1. / np.sqrt(Lambda_y))) * b
        zeta = U_y * zeta
        # g_opt = ?

        return zeta, float(r2)
        # pass

    # TODO: add plot methods