# -*- coding: utf-8 -*-
# @Author: Pan Chao
# @Date:   2017-12-26 10:19:32
# @Last Modified by:   Pan Chao
# @Last Modified time: 2017-12-26 10:20:11


import numpy as np
import scipy as sp
import scipy.linalg as slin
import sys, traceback
# import warnings
# import copy
from sklearn import linear_model

from .Data import Data, ParameterizedData, ParameterizedDataWithAdditiveKernel
from .Parameters import Parameters
from .Group import Group
from okgtreg.Kernel import Kernel


"""
X: covariate matrix (high dimensional)
y: response vector (univariate)
groupStructure: partition of variables into groups
"""


class OKGTReg(object):
    def __init__(self, data, parameters=None, eps=1e-6, kernel=None, group=None):
        """
        Two constructors:
            1) OKGTReg(data, parameters)
            2) OKGTReg(data, kernel, group)

        The second constructor assumes that all groups share the same kernel.

        :type data: Data
        :param data:

        :type parameters: Parameters or None
        :param parameters:

        :type eps: float
        :param eps: regularization coefficient for regularizing kernel matrices

        :type kernel: Kernel or None
        :param kernel:

        :rtype: OKGTReg
        :return:
        """
        # private field, not to access directly
        if parameters is None:
            try:
                parameters = Parameters(group, kernel, [kernel]*group.size)
                self.parameterizedData = ParameterizedData(data, parameters)
            except AttributeError:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("** The parameters \"kernel\" or \"group\" are not correctly specified. **")
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
        else:
            self.parameterizedData = ParameterizedData(data, parameters)

        self.eps = eps
        self.data = data

    def getX(self):
        return self.parameterizedData.X

    def getY(self):
        return self.parameterizedData.y

    def getGroupStructure(self):
        """

        :rtype: Group
        :return:
        """
        return self.parameterizedData.group

    def getKernels(self, yOrX):
        """

        :param yOrX:

        :rtype: Kernel or list of Kernel objects
        :return:
        """
        if yOrX is 'y':
            return self.parameterizedData.ykernel
        else:
            return self.parameterizedData.xkernels

    def getPartition(self):
        return self.parameterizedData.partition

    def getSampleSize(self):
        return self.parameterizedData.n

    def getGroupSize(self):
        return self.parameterizedData.groupSize

    def _train_Vanilla(self):
        n = self.getSampleSize()
        l = self.getGroupSize()

        # print "** Start OKGT Training (Vanilla) ... **"

        Rxx, Gx = self.parameterizedData.covarianceOperatorForX(returnAll=True)
        Ryy, Gy = self.parameterizedData.covarianceOperatorForY(returnAll=True)
        # Ryx = self.parameterizedData.crossCovarianceOperator()
        Ryx = Gy.dot(Gx.T) / n

        D, P = np.linalg.eigh(Ryy + self.eps * np.identity(n))
        D = D[::-1]
        P = P[:, ::-1]
        D_inv = np.diag(1. / np.sqrt(D))
        Gy_inv = D_inv.dot(P.T) # Ryy^{-1/2}

        Rxx_inv = np.linalg.inv(Rxx + self.eps * np.identity(n * l))

        #TODO: if Rxx is large, the inverse would be slow.
        VyxVxy = reduce(np.dot, [Gy_inv, Ryx, Rxx_inv, Ryx.T, Gy_inv.T])

        # g: optimal transformation for y
        r2, beta = slin.eigh(VyxVxy, eigvals=(n-1, n-1)) # only need the largest eigen value and vector
        _zeta = D_inv.dot(beta)
        zeta = P.dot(_zeta)
        g_opt = Gy.dot(zeta)

        # f: optimal transformation for x
        # TODO: use matrix multiplication to replace the following loop
        _x_i = Ryx.T.dot(g_opt)
        x_i = Rxx_inv.dot(_x_i)
        f_opt_ls = []
        for i in range(l):
            x_ii = x_i[i*n : (i+1)*n]
            Gx_i = Gx[i*n : (i+1)*n, :]
            f_i_opt = Gx_i.dot(x_ii)
            f_i_norm = np.sqrt(x_ii.T.dot(f_i_opt))
            f_i_opt = f_i_opt / f_i_norm
            f_opt_ls.append(f_i_opt)

        f_opt = np.column_stack(f_opt_ls)

        # print "** Success **"
        return dict(g=g_opt, f=f_opt, r2=float(r2))
        # Capture the coefficients
        # return dict(g=g_opt, f=f_opt, r2=float(r2), coef=x_i.reshape((n,l), order='F'))

    def _train_Nystroem(self, nComponents, seed=None):
        """
        Training an OKGT where the kernel matrices are approximated by low rank matrices
        using Nystroem method.

        :type nComponents: int
        :param nComponents: How many data points will be used to construct the mapping for
                            kernel matrix low rank approximation.

        :type seed: int, optional
        :param seed: seed for the random number generator for Nystroem method.

        :rtype: dict
        :return: optimal g, optimal f, and R2
        """

        n = self.getSampleSize()
        l = self.getGroupSize()

        # print "** Start OKGT Training (Nystroem) **"

        N0 = np.identity(n) - np.ones((n, n)) / n
        ykernel = self.getKernels('y')
        Gy = ykernel.gram_Nystroem(self.getY()[:, np.newaxis], nComponents, seed)
        Uy, Gy_s, Gy_V = np.linalg.svd(N0.dot(Gy), full_matrices=False)
        lambday = Gy_s**2
        my = len(Gy_s)

        Ux = []
        lambdax = []
        xkernels = self.getKernels('x')  # list of Kernels for X, one for each group
        for i in range(l):
            inds = [ind - 1 for ind in self.getPartition()[i]] # column index for i-th group
            Gi = xkernels[i].gram_Nystroem(self.getX()[:, inds], nComponents, seed)
            Ui, Gi_s, Gi_V = np.linalg.svd(N0.dot(Gi), full_matrices=False)
            Ux.append(Ui)
            lambdai = Gi_s**2
            lambdax.append(lambdai)

        lambdax_row = np.hstack(lambdax)
        Ux_row = np.hstack(Ux)
        Ux_diag = sp.sparse.block_diag(Ux)

        T = reduce(np.dot, [np.diag(lambday / (lambday + self.eps)), Uy.T, Ux_row, np.diag(lambdax_row)])
        R = np.diag((lambdax_row + self.eps)**2) + \
                reduce(np.dot, [np.diag(lambdax_row),
                               Ux_row.T.dot(Ux_row) - np.identity(len(lambdax_row)),
                               np.diag(lambdax_row)])
        R_inv = np.linalg.inv(R)  # much faster now
        vv = reduce(np.dot, [T, R_inv, T.T])

        eigval, eigvec = sp.linalg.eigh(vv, eigvals=(my-1, my-1))
        r2 = float(eigval)
        _g_opt = np.diag(lambday).dot(eigvec)
        g_opt = Uy.dot(_g_opt)

        _f_opt = np.diag(np.sqrt(lambday**2 + self.eps) * lambday).dot(eigvec)
        _f_opt = T.T.dot(_f_opt)
        _f_opt = R_inv.dot(_f_opt)
        _f_opt = np.diag(lambdax_row).dot(_f_opt)
        _f_opt = Ux_diag.dot(_f_opt)
        f_opt =  _f_opt.reshape((n, l), order='F')

        # print "** Success **"
        return dict(g=g_opt, f=f_opt, r2=r2)

    def train(self, method='vanilla', nComponents=None, seed=None):
        if method is 'nystroem' and nComponents is None:
            raise ValueError("** \"nComponent\" is not provided for \"nystroem\" method.")

        # Use Python dictionary to implement switch-case structure. Details can be found at:
        #   http://bytebaker.com/2008/11/03/switch-case-statement-in-python/
        trainFunctions = {'vanilla': self._train_Vanilla,
                          'nystroem': lambda: self._train_Nystroem(nComponents=nComponents, seed=seed)}

        try:
            return trainFunctions[method]()
        except KeyError:
            print("** Method \"%s\" could not be found. **" % method)


# The following OKGTReg2 class has the same methods as those in
# OKGTReg class. The difference lies in how the covariance
# operator of $R_{XX}$ and cross-covariance operator $R_{XY}$
# are constructed.
#
# In the OKGTReg class, since the centered gram matrices
# for X are stacked, $R_{XX}$ is of size (nd x nd), and $R_{XY}$
# is of size (nd x n), where $n$ is the sample size and $d$ is the
# number of groups in the given group structure. These
# expanded matrices increases the computation complexity of
# the vanilla algorithm, which is $O(n^3d^3)$.
#
# In the following implementation of OKGTReg2 class, the additive
# kernel for X is directly exploited. By using the kernel
#
#       $K_X = K_1 + K_2 + \ldots + K_d$,
#
# the gram matrix for $X$ is constructed as the sum (instead
# of block stack) of the individual gram matrices, which is
# of size n x n. As a result, the size of $R_{XX}$ and $R_{XY}$
# are both n x n. Thus, the time complexity of the algorithm
# becomes $O(n^3)$.
#
# Note that though given the evaluated gram matrices, the time
# complexity is $O(n^3)$, which is independent of the number
# of groups. The speed of evaluating gram matrices depend on
# the number of groups. Since the evaluation of a single gram
# matrix takes $O(n^2)$, it becomes $O(dn^2)$ for a group structure
# with $d$ groups.
class OKGTReg2(object):
    def __init__(self, data, parameters=None, eps=1e-6, kernel=None, group=None):
        """
        Two constructors:
            1) OKGTReg(data, parameters)
            2) OKGTReg(data, kernel, group)

        The second constructor assumes that all groups share the same kernel.

        :type data: Data
        :param data:

        :type parameters: Parameters or None
        :param parameters:

        :type eps: float
        :param eps: regularization coefficient for regularizing kernel matrices

        :type kernel: Kernel or None
        :param kernel:

        :rtype: OKGTReg
        :return:
        """
        # private field, not to access directly
        if parameters is None:
            try:
                parameters = Parameters(group, kernel, [kernel] * group.size)
                # self.parameterizedData = ParameterizedData(data, parameters)
                self.parameterizedData = ParameterizedDataWithAdditiveKernel(data, parameters)
            except AttributeError:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("** The parameters \"kernel\" or \"group\" are not correctly specified. **")
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
        else:
            # self.parameterizedData = ParameterizedData(data, parameters)
            self.parameterizedData = ParameterizedDataWithAdditiveKernel(data, parameters)

        self.eps = eps
        self.data = data

    def getX(self):
        return self.parameterizedData.X

    def getY(self):
        return self.parameterizedData.y

    def getGroupStructure(self):
        """

        :rtype: Group
        :return:
        """
        return self.parameterizedData.group

    def getKernels(self, yOrX):
        """

        :param yOrX:

        :rtype: Kernel or list of Kernel objects
        :return:
        """
        if yOrX is 'y':
            return self.parameterizedData.ykernel
        elif yOrX is 'x':
            return self.parameterizedData.xkernels
        else:
            raise ValueError("** [ERROR] the parameter value is not recognized! **")

    def getPartition(self):
        return self.parameterizedData.partition

    def getSampleSize(self):
        return self.parameterizedData.n

    def getGroupSize(self):
        return self.parameterizedData.groupSize

    def _train_Vanilla(self):
        n = self.getSampleSize()
        l = self.getGroupSize()

        # print "** Start OKGT Training (Vanilla)**"

        Rxx, Gx, Gx_list = self.parameterizedData.covarianceOperatorForX(returnAll=True)
        Ryy, Gy = self.parameterizedData.covarianceOperatorForY(returnAll=True)
        Ryx = Gy.dot(Gx.T) / n

        D, P = np.linalg.eigh(Ryy + self.eps * np.identity(n))
        D = D[::-1]
        P = P[:, ::-1]
        D_inv = np.diag(1. / np.sqrt(D))
        Gy_inv = D_inv.dot(P.T)  # Ryy^{-1/2}

        Rxx_inv = np.linalg.inv(Rxx + self.eps * np.identity(n))

        # TODO: if Rxx is large, the inverse would be slow.
        VyxVxy = reduce(np.dot, [Gy_inv, Ryx, Rxx_inv, Ryx.T, Gy_inv.T])

        # g: optimal transformation for y
        r2, beta = slin.eigh(VyxVxy, eigvals=(n - 1, n - 1))  # only need the largest eigen value and vector
        _zeta = D_inv.dot(beta)
        zeta = P.dot(_zeta)
        g_opt = Gy.dot(zeta)

        # f: optimal transformation for x
        # TODO: use matrix multiplication to replace the following loop
        _alpha_i = Ryx.T.dot(g_opt)
        alpha_i = Rxx_inv.dot(_alpha_i)
        f_opt_ls = []
        for i in range(l):
            f_i_opt = Gx_list[i].dot(alpha_i)
            f_i_norm = np.sqrt(alpha_i.T.dot(f_i_opt))
            f_i_opt = f_i_opt / f_i_norm
            f_opt_ls.append(f_i_opt)

        f_opt = np.column_stack(f_opt_ls)

        # print "** Success **"
        return dict(g=g_opt, f=f_opt, r2=float(r2))
        # Capture the coefficients:
        # return dict(g=g_opt, f=f_opt, r2=float(r2), coef=alpha_i)

    def _train_Nystroem(self, nComponents, seed=None):
        """
        Training an OKGT where the kernel matrices are approximated by low rank matrices
        using Nystroem method.

        :type nComponents: int
        :param nComponents: How many data points will be used to construct the mapping for
                            kernel matrix low rank approximation.

        :type seed: int, optional
        :param seed: seed for the random number generator for Nystroem method.

        :rtype: dict
        :return: optimal g, optimal f, and R2
        """

        n = self.getSampleSize()
        l = self.getGroupSize()

        # print "** Start OKGT Training (Nystroem) **"

        N0 = np.identity(n) - np.ones((n, n)) / n
        ykernel = self.getKernels('y')
        Gy = ykernel.gram_Nystroem(self.getY()[:, np.newaxis], nComponents, seed)
        Uy, Gy_s, Gy_V = np.linalg.svd(N0.dot(Gy), full_matrices=False)
        lambday = Gy_s ** 2
        my = len(Gy_s)

        Ux = []
        lambdax = []
        xkernels = self.getKernels('x')  # list of Kernels for X, one for each group
        for i in range(l):
            inds = [ind - 1 for ind in self.getPartition()[i]]  # column index for i-th group
            Gi = xkernels[i].gram_Nystroem(self.getX()[:, inds], nComponents, seed)
            Ui, Gi_s, Gi_V = np.linalg.svd(N0.dot(Gi), full_matrices=False)
            Ux.append(Ui)
            lambdai = Gi_s ** 2
            lambdax.append(lambdai)

        lambdax_row = np.hstack(lambdax)
        Ux_row = np.hstack(Ux)
        Ux_diag = sp.sparse.block_diag(Ux)

        T = reduce(np.dot, [np.diag(lambday / (lambday + self.eps)), Uy.T, Ux_row, np.diag(lambdax_row)])
        R = np.diag((lambdax_row + self.eps) ** 2) + \
            reduce(np.dot, [np.diag(lambdax_row),
                            Ux_row.T.dot(Ux_row) - np.identity(len(lambdax_row)),
                            np.diag(lambdax_row)])
        R_inv = np.linalg.inv(R)  # much faster now
        vv = reduce(np.dot, [T, R_inv, T.T])

        eigval, eigvec = sp.linalg.eigh(vv, eigvals=(my - 1, my - 1))
        r2 = float(eigval)
        _g_opt = np.diag(lambday).dot(eigvec)
        g_opt = Uy.dot(_g_opt)

        _f_opt = np.diag(np.sqrt(lambday ** 2 + self.eps) * lambday).dot(eigvec)
        _f_opt = T.T.dot(_f_opt)
        _f_opt = R_inv.dot(_f_opt)
        _f_opt = np.diag(lambdax_row).dot(_f_opt)
        _f_opt = Ux_diag.dot(_f_opt)
        f_opt = _f_opt.reshape((n, l), order='F')

        # print "** Success **"
        return dict(g=g_opt, f=f_opt, r2=r2)

    def train(self, method='vanilla', nComponents=None, seed=None):
        if method is 'nystroem' and nComponents is None:
            raise ValueError("** \"nComponent\" is not provided for \"nystroem\" method.")

        # Use Python dictionary to implement switch-case structure. Details can be found at:
        #   http://bytebaker.com/2008/11/03/switch-case-statement-in-python/
        trainFunctions = {'vanilla': self._train_Vanilla,
                          'nystroem': lambda: self._train_Nystroem(nComponents=nComponents, seed=seed)}

        try:
            return trainFunctions[method]()
        except KeyError:
            print("** Method \"%s\" could not be found. **" % method)

    def _train_Vanilla2(self, h):
        """
        OKGT fitting with known response function. That is,
        the kernel for y is fixed to be linear (inner-product)
        kernel.

        :param h: the fixed response or fixed transformation of
                  the response.
        :return:
        """
        # print "** Start OKGT Training (Vanilla) with Known h **"

        n = self.getSampleSize()
        l = self.getGroupSize()

        Rxx, Gx, Gx_list = self.parameterizedData.covarianceOperatorForX(returnAll=True)
        # Since we assume the response transformation is known and
        # given by h, the kernel for h is fixed to be linear.
        yKernel = Kernel('linear')
        Gy = yKernel.gram(h[:, np.newaxis])
        Ryy = Gy.dot(Gy.T) / n
        Ryx = Gy.dot(Gx.T) / n

        D, P = np.linalg.eigh(Ryy + self.eps * np.identity(n))
        D = D[::-1]
        P = P[:, ::-1]
        D_inv = np.diag(1. / np.sqrt(D))
        Gy_inv = D_inv.dot(P.T)  # Ryy^{-1/2}

        Rxx_inv = np.linalg.inv(Rxx + self.eps * np.identity(n))

        # TODO: if Rxx is large, the inverse would be slow.
        VyxVxy = reduce(np.dot, [Gy_inv, Ryx, Rxx_inv, Ryx.T, Gy_inv.T])

        # g: optimal transformation for y
        ## only need the largest eigen value and vector
        r2, beta = slin.eigh(VyxVxy, eigvals=(n - 1, n - 1))
        _zeta = D_inv.dot(beta)
        zeta = P.dot(_zeta)
        gval = Gy.dot(zeta)  # discrete

        # Note: Since this training function is for the case when the response
        # transformation is known (given by h), there is no need to estimate
        # the function of g.

        # f: optimal transformation for x
        ## TODO: use matrix multiplication to replace the following loop
        _alpha_i = Ryx.T.dot(gval)
        alpha_i = Rxx_inv.dot(_alpha_i)
        fval_ls = []  # to collect predicted values

        ## collect transformation functions as callables
        ## and the predicted values
        xKernelList = self.getKernels('x')
        f_opt_callable_dict = {}
        for i in range(l):
            # TODO: there are normlization and centering issue in the
            # todo: callable functions. Need to be addressed before they
            # todo: be used to calculated the predicted value.
            ### Construct transformation functions as callables
            #### In the following, self.parameterizedData.getXFromGroup(i+1)
            #### returns a 2d array even for univariate group.
            #### A 2d array is the required type for the first argument of kernelSpan
            xkernelSpan = xKernelList[i].kernelSpan(self.parameterizedData.getXFromGroup(i + 1), alpha_i)
            # ### Centering (speed killer)
            # xgrams_nocenter_list = self.parameterizedData._getGramsForX(centered=False)
            # offset = ( xgrams_nocenter_list[i] * alpha_i[:, np.newaxis] ).sum() / n
            # f_opt_callable_dict[i+1] = (lambda x: xkernelSpan(x) - offset)
            f_opt_callable_dict[i + 1] = xkernelSpan

            ### The predicted value for the currrent data set
            f_i_opt = Gx_list[i].dot(alpha_i)
            f_i_norm = np.sqrt(alpha_i.T.dot(f_i_opt))
            f_i_opt = f_i_opt / f_i_norm  # discrete
            fval_ls.append(f_i_opt)
        fval = np.column_stack(fval_ls)

        # print "** Success **"

        # return dict(g=g_opt, f=f_opt, r2=float(r2))
        return dict(g=gval, f=fval, r2=float(r2), f_call=f_opt_callable_dict)

    def _train_lr(self, h):
        '''
        Since we know the true response, we can use least square
        regression to estimate the transformations for the covariate
        groups.

        :type h: 1d array
        :param h: known response transformation
        :return:
        '''
        n = self.getSampleSize()
        l = self.getGroupSize()

        # # normalize and center h
        # h = h - np.mean(h)
        # h = h / np.linalg.norm(h, ord=2)

        # construct the additive kernel matrix,
        # all component matrices are NOT CENTERED
        Kx_list = self.parameterizedData._getGramsForX(centered=False)
        Kx_add = sum(Kx_list)
        # Linear regression (with intercept)
        ## Reference:
        ## http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        # clf = linear_model.LinearRegression(fit_intercept=False)
        clf = linear_model.LinearRegression(fit_intercept=True)
        clf.fit(Kx_add + self.eps * np.identity(n), h)  # TODO: over-fitting?
        # clf.fit(Kx_add, h)  # TODO: over-fitting?
        a0 = clf.intercept_
        alpha = clf.coef_
        # construct covariate transformations
        f_est_list = [Kx_list[j].dot(alpha) for j in range(l)]
        # calculate R2
        h_hat = sum(f_est_list) + a0
        r2 = 1 - sum((h - h_hat) ** 2) / sum((h - np.mean(h)) ** 2)  # usual R2 for LR

        # Collect transformation functions as callables
        xKernelList = self.getKernels('x')
        f_callable_dict = {}
        f_callable_dict[0] = a0  # save intercept
        for i in range(l):
            # each f_j is a kernel span
            xkernelSpan = xKernelList[i].kernelSpan(self.parameterizedData.getXFromGroup(i + 1), alpha)
            f_callable_dict[i + 1] = xkernelSpan

        return dict(g=h, f=np.column_stack(f_est_list), r2=r2, f_call=f_callable_dict)



# TODO: For optimal split and merge methods, they can either using the existing kernel
# todo: or a new kennel function. The current implementation provides a new kernel. An
# todo: alternative is to use the kernel in the parameters if all groups share the same
# todo: kernel function.
class OKGTRegForDetermineGroupStructure(OKGTReg):
    def __init__(self, data, parameters, eps=1e-6):
        OKGTReg.__init__(self, data, parameters, eps)
        self.f = None
        self.g = None
        self.r2 = None
        self.bestR2 = None

    def train(self, method='vanilla', nComponents=None, seed=None):
        if method is 'nystroem' and nComponents is None:
            raise ValueError("** \"nComponent\" is not provided for \"nystroem\" method.")

        # Use Python dictionary to implement switch-case structure. Details can be found at:
        #   http://bytebaker.com/2008/11/03/switch-case-statement-in-python/
        trainFunctions = {'vanilla': self._train_Vanilla,
                          'nystroem': lambda: self._train_Nystroem(nComponents=nComponents, seed=seed)}

        try:
            fit = trainFunctions[method]()
            self.f = fit['f']
            self.g = fit['g']
            self.r2 = fit['r2']
            return
        except KeyError:
            print("** Method \"%s\" could not be found. **" % method)

    def optimalSplit(self, kernel, method='vanilla', nComponents=None, seed=None, threshold=0.):
        """
        Given the current group structure, we attempt to completely split each multi-variate
        group one-by-one and the corresponding OKGT is fitted. If splitting a group improves
        the estimated R2, then it is considered as an optimal group to split. The optimal group
        to split is determined to be the one which improves R2 the most.

        For example, for the current group structure ([1], [2,3], [4,5,6], [7]), each of the
        two multi-variate groups [2,3] and [4,5,6] is completely split into ([2], [3]) and
        ([4], [5], [6]). Each of the resulting OKGT is trained and the corresponding R2's are
        obtained. If splitting [2,3] results in a higher improvement in R2 than splitting [4,5,6],
        then splitting [2,3] is considered to be more optimal, and the function returns
        ([1], [2], [3], [4,5,6], [7]) as the optimal group structure. If splitting [4,5,6] is more
        optimal, then it returns ([1], [2,3], [4], [5], [6], [7]). In the worst case, if splitting
        either group results in no improvement (i.e. R2 does not increase), then no splitting is
        applied and the current group structure is returned.

        **Note: the current split procedure is very aggressive. The covariates in the chosen group
        are separated into univariate groups.** See `optimalSplit2` for a less aggressive implementation.

        :type kernel: Kernel
        :param kernel: kernel function for each grouped transformation.

        :type method: string
        :param method: OKGT training method, "vanilla" or "nystroem".

        :type nComponents: int
        :param nComponents: sample size used for Nystroem low-rank approximation.

        :type seed: int
        :param seed: seed used for the random number generator of Nystroem low-rank approximation.

        :type threshold: float, >=0
        :param threshold: if the improvement in R2 by splitting a group  exceeds this threshold,
                          it is considered significant and the split is performed.

        :rtype: OKGTRegForDetermineGroupStructure
        :return: optimal group structure encapsulated in an OKGTRegForDetermineGroupStructure object
                 determined the optimal split procedure.
        """
        # Check if current OKGT is already trained
        if self.r2 is None:
            self.train(method=method, nComponents=nComponents, seed=seed)

        if self.getGroupSize() == self.parameterizedData.p:  # all univariate groups in the structure
            # warnings.warn("** All groups are univariate. No need to split. **")
            print "** All groups are univariate. No need to split. **"
            return self
        else:  # start splitting attempts
            bestOkgt = self
            bestR2 = self.r2
            print("** Current group structure: %s, R2 = %.04f. **\n" % (bestOkgt.getGroupStructure(), bestR2))

            currentGroup = self.getGroupStructure()
            for i in np.arange(currentGroup.size) + 1:
                if len(currentGroup.getPartitions(i)[0]) > 1:
                    newGroup = currentGroup._splitOneGroup(i)
                    newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
                    newOkgt = OKGTRegForDetermineGroupStructure(self.data, newParameters)
                    newOkgt.train(method=method, nComponents=nComponents, seed=seed)
                    print("** Tested group structure by complete split: "
                          "%s, R2 = %.04f. **" % (newGroup, newOkgt.r2))
                    # if newOkgt.r2 > bestR2:
                    if newOkgt.r2 - bestR2 > threshold:
                        bestR2 = newOkgt.r2
                        bestOkgt = newOkgt
                        print("** Improving -> Better group structure: "
                              "%s, R2 = %.04f. **" % (newGroup, bestR2))

            if self.getGroupStructure() == bestOkgt.getGroupStructure():
                # warnings.warn("** No split can improve R2. **")
                print "\n** No split can improve R2. **\n"
                return self
            else:
                print("\n** New group structure after optimal split: "
                      "%s, R2 = %.04f. **\n" % (bestOkgt.getGroupStructure(), bestR2))
                return bestOkgt

    def optimalSplit2(self, kernel, method='vanilla', nComponents=None, seed=None,
                      threshold=0., maxSplit=1):
        """
        A less aggressive split procedure. That is, the returned group structure is
        given by random split (one or multiple covariates in a group) instead of complete
        split. However, during the intermediate steps, the improvement of R2 is based
        on the complete split.

        :type kernel: Kernel
        :param kernel: kernel function used to train OKGT during searching for optimal
                       group structure after a less aggressive split procedure.

        :type method: str
        :param method: 'vanilla' or 'nystroem'

        :type nComponents: int
        :param nComponents: number of components for Nystroem low rank approximation.
                            Details can be found at `Scikit-learn Nystroem <http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem>`_

        :type seed: int
        :param seed: seeding the random number generator for Nystroem low rank approximation

        :type threshold: float, >=0
        :param threshold: if the improvement in R2 by splitting a group exceeds this threshold,
                          it is considered significant and the split is performed.

        :type maxSplit: int
        :param maxSplit: the maximum number of covariates to be split into univariate groups
                         in a chosen group.

        :rtype: OKGTRegForDetermineGroupStructure
        :return:
        """
        if self.r2 is None:
            self.train(method=method, nComponents=nComponents, seed=seed)

        if self.getGroupSize() == self.parameterizedData.p:
            print "** All groups are univariate. No need to split. **"
            self.bestR2 = self.r2
            return self
        else:  # start splitting attempts
            bestOkgt = self
            bestR2 = self.r2  # updated based on complete split
            print("** Current group structure: %s, R2 = %.04f. **\n" % (bestOkgt.getGroupStructure(), bestR2))

            improved = False
            # Update group structure
            currentGroup = self.getGroupStructure()
            for i in np.arange(currentGroup.size) + 1:
                len_i = len(currentGroup[i])  # i-th part of the group structure
                if len_i > 1:
                    testGroup = currentGroup.split(i)  # completely split i-th group
                    testParameters = Parameters(testGroup, kernel, [kernel]*testGroup.size)
                    testOkgt = OKGTRegForDetermineGroupStructure(self.data, testParameters)
                    testOkgt.train(method=method, nComponents=nComponents, seed=seed)
                    print("** Tested group structure by complete split: "
                          "%s, R2 = %.04f. **" % (testOkgt.getGroupStructure(), testOkgt.r2))
                    # Thresholding R2 improvement
                    if testOkgt.r2 - bestR2 > threshold:
                        improved = True
                        bestR2 = testOkgt.r2
                        # Randomly split one or multiple covariate from the
                        # current group structure (less aggressive)
                        if maxSplit > len_i:
                            newGroup = testGroup
                        else:
                            newGroup = currentGroup.split(i, True, seed, maxSplit)
                        print("** Improving! -> Update by random split: %s. **" % newGroup)
                    else:
                        print("** No improving. **")

            # Return result
            if improved:
                print("\n** New group structure after optimal random split: %s. **\n" % newGroup)
                bestParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
                bestOkgt = OKGTRegForDetermineGroupStructure(self.data, bestParameters)
                bestOkgt.train(method=method, nComponents=nComponents, seed=seed)
                bestOkgt.bestR2 = bestR2
                return bestOkgt
                # return newOkgt
            else: # no improvement
                print "\n** No split can improve R2. **\n"
                self.bestR2 = self.r2
                return self

    def optimalMerge(self, kernel, method='vanilla', nComponents=None, seed=None, threshold=0.):
        """
        Determine the optimal group structure by merging groups in
        the current group structure. The merging is performed for
        every pair of groups provided that at least one of the group
        is univariate. The reason for not merging every pair of the
        groups is to reduce the computational burden of the procedure.

        :param kernel:

        :type method: str
        :param method: `vanilla` or `nystroem`

        :type nComponents: int
        :param nComponents: number of components for Nystroem low rank
                            approximation. Details of the Nystroem method
                            can be found at
                            `Scikit-learn Nystroem <http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem>`_

        :type seed: int
        :param seed: seeding the random number generator for Nystroem
                     low rank approximation

        :type threshold: float, >=0
        :param threshold: if the improvement in R2 by merging two groups
                          exceeds this threshold, it is considered
                          significant and the merge is performed.

        :rtype: OKGTRegForDetermineGroupStructure
        :return:
        """

        # Check if current OKGT is already trained
        if self.r2 is None:
            self.train(method=method, nComponents=nComponents, seed=seed)

        if self.getGroupSize() == 1:  # only one group in the structure
            print("** There is only one group. No need to merge. **")
            # warnings.warn("** There is only one group. No need to merge. **")
            self.bestR2 = self.r2
            return self
        elif not any(len(part)==1 for part in self.getPartition()):
            print("** All groups are multi-variate. No merge. **")
            self.bestR2 = self.r2
            return self
        else:  # start merging attempts
            bestR2 = self.r2
            bestOkgt = self
            print("** Current group structure: %s, R2 = %.04f. **" % (bestOkgt.getGroupStructure(), bestR2))

            improved = False
            # Try to merger two groups
            currentGroup = self.getGroupStructure()
            for i in np.arange(1, self.getGroupSize()):  # excluding the last group
                for j in np.arange(i+1, self.getGroupSize()+1):
                    # only attempt to merge a univariate group with another one
                    if len(currentGroup.getPartitions(i)[0]) > 1 and len(currentGroup.getPartitions(j)[0]) > 1:
                        continue
                    else:
                        newGroup = currentGroup._mergeTwoGroups(i, j)
                        # print newGroup
                        newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
                        newOkgt = OKGTRegForDetermineGroupStructure(self.data, newParameters)
                        newOkgt.train(method=method, nComponents=nComponents, seed=seed)
                        # print newOkgt.r2
                        # if newOkgt.r2 > bestR2:
                        if newOkgt.r2 - bestR2 > threshold:
                            improved = True
                            bestR2 = newOkgt.r2
                            bestOkgt = newOkgt
                            print("** Better group structure: %s, R2 = %.04f. **" % (newGroup, bestR2))

            if improved:
                print("\n** New group structure after optimal merge: "
                      "%s, R2 = %.04f. **\n" % (bestOkgt.getGroupStructure(), bestR2))
                bestOkgt.bestR2 = bestR2
                return bestOkgt
            else:
                print "\n** No merge can improve R2. **\n"
                self.bestR2 = self.r2
                return self