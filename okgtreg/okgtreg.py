import numpy as np
import scipy as sp
import scipy.linalg as slin
import warnings
import copy

# from .Data import *
# from .Kernel import *

from .Data import Data, ParameterizedData
from .Parameters import Parameters
from .Group import Group


"""
X: covariate matrix (high dimensional)
y: response vector (univariate)
groupStructure: partition of variables into groups
"""


class OKGTReg(object):
    def __init__(self, data, parameters, eps=1e-6):
        """

        :type data: Data
        :param data:

        :type params: Parameters
        :param params:

        :rtype: OKGTReg
        :return:
        """
        # private field, not to accessed directly
        self.parameterizedData = ParameterizedData(data, parameters)
        self.eps = eps
        self.data = data
        # To be updated after calling .train()
        # self.f = None
        # self.g = None
        # self.r2 = None

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

        # print "** Start OKGT Training (Vanilla)**"

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
        # self.f = f_opt
        # self.g = g_opt
        # self.r2 = float(r2)
        # return

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
        # self.f = f_opt
        # self.g = g_opt
        # self.r2 = r2
        # return

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

    # # Assume all groups share the same kernel function
    # def optimalSplit(self, kernel, method='vanilla', nComponents=None, seed=None):
    #     if self.getGroupSize() == self.parameterizedData.p:
    #         warnings.warn("** All groups are univariate. No need to split. **")
    #         return self
    #     else:
    #         # Train OKGT for the current group structure
    #         res = self.train(method=method, nComponents=nComponents, seed=seed)
    #         bestR2 = res['r2']
    #         bestOkgt = self
    #         print("** Current group structure: %s, R2 = %.04f. **" % (bestOkgt.getGroupStructure(), bestR2))
    #
    #         # For each possible split, train the corresponding OKGT.
    #         # Note it is possible that the current group structure is still the best.
    #         currentGroup = self.getGroupStructure()
    #         for i in np.arange(currentGroup.size) + 1:
    #             if len(currentGroup.getPartition(i)) > 1:
    #                 newGroup = currentGroup._splitOneGroup(i)
    #                 newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
    #                 newOkgt = OKGTReg(self.data, newParameters)  # create a new OKGTReg object
    #                 res = newOkgt.train(method=method, nComponents=nComponents, seed=seed)
    #                 if res['r2'] > bestR2:
    #                     bestR2 = res['r2']
    #                     bestOkgt = newOkgt
    #                     print("** Better group structure: %s, R2 = %.04f. **" % (newGroup, bestR2))
    #
    #         print("Group structure after optimal split: %s, R2 = %.04f." % (bestOkgt.getGroupStructure(), bestR2))
    #         return bestOkgt
    #
    # # Assume all groups share the same kernel function
    # def optimalMerge(self, kernel, method='vanilla', nComponents=None, seed=None):
    #     """
    #     Combine two groups which results in the most improvement in OKGT fitting.
    #
    #     :return:
    #     """
    #     if self.getGroupSize() == 1:
    #         warnings.warn("** There is only one group. No need to merge. **")
    #         return self
    #     else:
    #         # print self.getGroupStructure()
    #         # Train OKGT for the current group structure
    #         res = self.train(method=method, nComponents=nComponents, seed=seed)
    #         bestR2 = res['r2']
    #         bestOkgt = self
    #         print("** Current group structure: %s, R2 = %.04f. **" % (bestOkgt.getGroupStructure(), bestR2))
    #
    #         # Try to merger two groups
    #         currentGroup = self.getGroupStructure()
    #         for i in np.arange(1, self.getGroupSize()):  # excluding the last group
    #             for j in np.arange(i+1, self.getGroupSize()+1):
    #                 # only attempt to merge a univariate group with another one
    #                 if len(currentGroup.getPartition(i)) > 1 and len(currentGroup.getPartition(j)) > 1:
    #                     continue
    #                 else:
    #                     newGroup = currentGroup._mergeTwoGroups(i, j)
    #                     newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
    #                     newOkgt = OKGTReg(self.data, newParameters)
    #                     res = newOkgt.train(method=method, nComponents=nComponents, seed=seed)
    #                     if res['r2'] > bestR2:
    #                         bestR2 = res['r2']
    #                         bestOkgt = newOkgt
    #                         print("** Better group structure: %s, R2 = %.04f. **" % (newGroup, bestR2))
    #
    #         print("Group structure after optimal merge: %s, R2 = %.04f." % (bestOkgt.getGroupStructure(), bestR2))
    #         return bestOkgt


class OKGTRegForDetermineGroupStructure(OKGTReg):
    def __init__(self, data, parameters, eps=1e-6):
        OKGTReg.__init__(self, data, parameters, eps)
        self.f = None
        self.g = None
        self.r2 = None

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

    def optimalSplit(self, kernel, method='vanilla', nComponents=None, seed=None):
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
            print("** Current group structure: %s, R2 = %.04f. **" % (bestOkgt.getGroupStructure(), bestR2))

            currentGroup = self.getGroupStructure()
            for i in np.arange(currentGroup.size) + 1:
                if len(currentGroup.getPartition(i)) > 1:
                    newGroup = currentGroup._splitOneGroup(i)
                    newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
                    newOkgt = OKGTRegForDetermineGroupStructure(self.data, newParameters)
                    newOkgt.train(method=method, nComponents=nComponents, seed=seed)
                    if newOkgt.r2 > bestR2:
                        bestR2 = newOkgt.r2
                        bestOkgt = newOkgt
                        print("** Better group structure: %s, R2 = %.04f. **" % (newGroup, bestR2))

            if self.getGroupStructure() == bestOkgt.getGroupStructure():
                # warnings.warn("** No split can improve R2. **")
                print "** No split can improve R2. **"
                return self
            else:
                print("New group structure after optimal split: "
                      "%s, R2 = %.04f." % (bestOkgt.getGroupStructure(), bestR2))
                return bestOkgt

    def optimalSplit2(self, kernel, method='vanilla', nComponents=None, seed=None):
        """
        A less aggressive split procedure, comparing to `optimalSplit`.

        :return:
        """
        if self.r2 is None:
            self.train(method=method, nComponents=nComponents, seed=seed)

        if self.getGroupSize() == self.parameterizedData.p:
            print "** All groups are univariate. No need to split. **"
            return self
        else:  # start splitting attempts
            bestOkgt = self
            bestR2 = self.r2
            print("** Current group structure: %s, R2 = %.04f. **" % (bestOkgt.getGroupStructure(), bestR2))

            currentGroup = self.getGroupStructure()
            for i in np.arange(currentGroup.size) + 1:
                if len(currentGroup.getPartition(i)) > 1:
                    newGroup = currentGroup._splitOneGroup(i)
                    newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
                    newOkgt = OKGTRegForDetermineGroupStructure(self.data, newParameters)
                    newOkgt.train(method=method, nComponents=nComponents, seed=seed)
                    if newOkgt.r2 > bestR2:
                        # randomly split one covariate from the current group
                        pass
        pass

    def optimalMerge(self, kernel, method='vanilla', nComponents=None, seed=None):
        # Check if current OKGT is already trained
        if self.r2 is None:
            self.train(method=method, nComponents=nComponents, seed=seed)

        if self.getGroupSize() == 1:  # only one group in the structure
            print "** There is only one group. No need to merge. **"
            # warnings.warn("** There is only one group. No need to merge. **")
            return self
        else:  # start merging attempts
            bestR2 = self.r2
            bestOkgt = self
            print("** Current group structure: %s, R2 = %.04f. **" % (bestOkgt.getGroupStructure(), bestR2))

            # Try to merger two groups
            currentGroup = self.getGroupStructure()
            for i in np.arange(1, self.getGroupSize()):  # excluding the last group
                for j in np.arange(i+1, self.getGroupSize()+1):
                    # only attempt to merge a univariate group with another one
                    if len(currentGroup.getPartition(i)) > 1 and len(currentGroup.getPartition(j)) > 1:
                        continue
                    else:
                        newGroup = currentGroup._mergeTwoGroups(i, j)
                        # print newGroup
                        newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
                        newOkgt = OKGTRegForDetermineGroupStructure(self.data, newParameters)
                        newOkgt.train(method=method, nComponents=nComponents, seed=seed)
                        # print newOkgt.r2
                        if newOkgt.r2 > bestR2:
                            bestR2 = newOkgt.r2
                            bestOkgt = newOkgt
                            print("** Better group structure: %s, R2 = %.04f. **" % (newGroup, bestR2))

            if self.getGroupStructure() == bestOkgt.getGroupStructure():
                print "** No merge can improve R2. **"
                # warnings.warn("** No merge can improve R2. **")
                return self
            else:
                print("New group structure after optimal merge: "
                      "%s, R2 = %.04f." % (bestOkgt.getGroupStructure(), bestR2))
                return bestOkgt
