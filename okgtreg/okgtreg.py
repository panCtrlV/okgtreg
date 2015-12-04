import numpy as np
import scipy as sp
import scipy.linalg as slin
import warnings

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
            print("** Method \"%s\" could not be found. **")

    # Assume all groups share the same kernel function
    def optimalSplit(self, kernel, method='vanilla', nComponents=None, seed=None):
        if self.getGroupSize() == self.parameterizedData.p:
            warnings.warn("** All groups are univariate. No need to split. **")
            return self
        else:
            # Train OKGT for the current group structure
            res = self.train(method=method, nComponents=nComponents, seed=seed)
            bestR2 = res['r2']
            bestOkgt = self
            print("** Current group structure: %s, R2 = %.04f. **" % (bestOkgt.getGroupStructure(), bestR2))

            # For each possible split, train the corresponding OKGT.
            # Note it is possible that the current group structure is still the best.
            currentGroup = self.getGroupStructure()
            for i in np.arange(currentGroup.size) + 1:
                if len(currentGroup.getPartition(i)) > 1:
                    newGroup = currentGroup._splitOneGroup(i)
                    newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
                    newOkgt = OKGTReg(self.data, newParameters)  # create a new OKGTReg object
                    res = newOkgt.train(method=method, nComponents=nComponents, seed=seed)
                    if res['r2'] > bestR2:
                        print("** New best group structure: %s, R2 = %.04f. **" % (newGroup, res['r2']))
                        bestR2 = res['r2']
                        bestOkgt = newOkgt

            print("Group structure after optimal split: %s, R2 = %.04f." % (bestOkgt.getGroupStructure(), bestR2))
            return bestOkgt

    # Assume all groups share the same kernel function
    def optimalMerge(self, kernel, method='vanilla', nComponents=None, seed=None):
        """
        Combine two groups which results in the most improvement in OKGT fitting.

        :return:
        """
        if self.getGroupSize() == 1:
            warnings.warn("** There is only one group. No need to merge. **")
            return self
        else:
            print self.getGroupStructure()
            # Train OKGT for the current group structure
            res = self.train(method=method, nComponents=nComponents, seed=seed)
            bestR2 = res['r2']
            bestOkgt = self
            print("** Current group structure: %s, R2 = %.04f. **" % (bestOkgt.getGroupStructure(), bestR2))

            # Try to merger two groups
            currentGroup = self.getGroupStructure()
            for i in np.arange(1, self.getGroupSize()):  # excluding the last group
                for j in np.arange(i+1, self.getGroupSize()+1):
                    if len(currentGroup.getPartition(i)) > 1 and len(currentGroup.getPartition(j)) > 1:
                        continue
                    else:
                        newGroup = currentGroup._mergeTwoGroups(i, j)
                        newParameters = Parameters(newGroup, kernel, [kernel]*newGroup.size)
                        newOkgt = OKGTReg(self.data, newParameters)
                        res = newOkgt.train(method=method, nComponents=nComponents, seed=seed)
                        if res['r2'] > bestR2:
                            print("** New best group structure: %s, R2 = %.04f. **" % (newGroup, res['r2']))
                            bestR2 = res['r2']
                            bestOkgt = newOkgt

            print("Group structure after optimal merge: %s, R2 = %.04f." % (bestOkgt.getGroupStructure(), bestR2))
            return bestOkgt

