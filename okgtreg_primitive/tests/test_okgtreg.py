__author__ = 'panc'

from okgtreg_primitive.okgtreg import *
from okgtreg_primitive.simulate import *
import okgtreg_primitive.kernel_selector as ks

import unittest
import numpy as np
import scipy.spatial.distance as distance
import scipy as sp
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

# Write lines longer than 80 chars in output file
# http://stackoverflow.com/questions/4286544/write-lines-longer-than-80-chars-in-output-file-python
np.set_printoptions(linewidth=300)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def is_symmetric(x):
    return np.all(x == x.T)

class TestOKGTReg(unittest.TestCase):
    def setUp(self):
        self.n = 500
        self.p = 1
        np.random.seed(10)
        self.y, self.x = SimData_Breiman1(self.n)

    # def test_constructKernelFns(self):
    #     kern1 = OKGTReg.ConstructKernelFns('Gaussian', {'sigam':0.3})
    #     kern2 = OKGTReg.ConstructKernelFns(['Laplace'], [{'alpha':5}])
    #     kern3 = OKGTReg.ConstructKernelFns(['Gaussian', 'Laplace', 'Exponential', 'Polynomial', 'Sigmoid'],
    #                                        [{'sigma':0.5}, {'alpha':1}, {'sigma':0.01},
    #                                         {'intercept':1, 'slope':2.3, 'degree':4}, {'a':1.2, 'r':10}])
    #     return (kern1, kern2, kern3)

    # def test_GramMatrix_symmetry(self):
    #     # y, x = SimData_Breiman1(n=500)
    #     kernFn = ks.KernelSelector('Gaussian', sigma=0.5)
    #     yGram = OKGTReg.GramMatrix(self.y, kernFn)
    #     # xGram = OKGTReg.GramMatrix(self.x, kernFn, centered=False)
    #     xGram = OKGTReg.GramMatrix(self.x, kernFn)
    #     self.assertTrue(is_symmetric(yGram))
    #     self.assertTrue(is_symmetric(xGram))
    #     # print np.linalg.eigvalsh(xGram)
    #     # xGramL = np.linalg.cholesky(xGram)
    #     # print xGramL
    #     # self.assertTrue(is_pos_def(yGram))
    #     # self.assertTrue(is_pos_def(xGram))
    #     # print(np.linalg.eigvals(xGram))
    #     # print (xGram==xGram.T)[1:5, 1:5]
    #     # print xGram[1:10, 1:10]
    #     # self.assertTrue(is_symmetric(xGram))
    #     # self.assertTrue(is_pos_def(xGram))
    #
    #     # I = np.identity(self.n)
    #     # II = np.matrix(np.ones((self.n, self.n)))
    #     # III = I - II/self.n
    #     # self.assertTrue(is_symmetric(III))
    #
    #     # self.assertTrue(is_symmetric(III * xGram * III.T))
    #     # IV = III * xGram * III.T
    #     # IV = (IV + IV.T)/2.
    #     # col_mean = xGram.mean(0)
    #     # row_mean = xGram.mean(1)
    #     # print row_mean.shape
    #     # print col_mean.shape
    #     # IV = xGram - col_mean - col_mean.T
    #     # self.assertTrue(is_symmetric(IV))
    #     # IV = xGram - col_mean - col_mean.T + xGram.sum()/self.n/self.n
    #
    #     # print (IV==IV.T)[1:20, 1:20]
    #     # print IV[1:20, 1:20]
    #     # print IV[IV!=IV.T].shape
    #     # self.assertTrue(is_symmetric(IV))
    #
    #     # distMatrix = distance.pdist(self.x, kernFn)
    #     # print distMatrix.shape
    #     # distMatrix = distance.squareform(distMatrix)
    #     # np.fill_diagonal(distMatrix, 1.)
    #     # print (distMatrix==distMatrix.T)[1:5, 1:5]
    #     # print distMatrix[1:5, 1:5]
    #     # self.assertTrue(is_symmetric(distMatrix))
    #
    #     # print '\n'
    #     # xGram2 = np.matrix(pairwise_distances(self.x, metric=kernFn))
    #     # print xGram2[1:5, 1:5]
    #     # print (xGram==xGram.T)[1:10, 1:10]
    #     # print np.linalg.eigvals(xGram)
    #     # self.assertTrue(is_pos_def(xGram))
    #
    #     # xGram2 = np.matrix(sp.linalg.sqrtm(xGram2 * xGram2.T))
    #     # print np.real(xGram2)[1:5, 1:5]
    #     # self.assertTrue(is_pos_def(np.real(xGram2)))
    #
    #     # u,s,v = np.linalg.svd(xGram2)
    #     # print '\n'
    #     # print s
    #     # print '\n'
    #     # print u
    #     # print '\n'
    #     # print v
    #     # xGram2 = u * np.diag(s) * u.T
    #     # print '\n'
    #     # print xGram2[1:5, 1:5]
    #     # self.assertTrue(is_pos_def(xGram2))

    # def test_GramMatrix_positive_definite(self):
    #     kernFn = ks.KernelSelector('Gaussian', sigma=0.5)
    #     yGram = OKGTReg.GramMatrix(self.y, kernFn)
    #     xGram = OKGTReg.GramMatrix(self.x, kernFn)
    #     self.assertTrue(is_pos_def(yGram + 0.001*np.identity(self.n)))
    #     self.assertTrue(is_pos_def(xGram + 0.001*np.identity(self.n)))
    #
    # def test_CovOperator_directSum_one_dimensional(self):
    #     kernelList = OKGTReg.ConstructKernelFns('Gaussian', {'sigma':0.5}) # one kernel for one dimension
    #     R, colStackBlockMatrix = OKGTReg.CovOperator_directSum(self.x, kernelList) # no need for group info for one dimension
    #     self.assertTrue( R.shape == (self.n, self.n) ) # check the correctness of dimension
    #     self.assertTrue( colStackBlockMatrix.shape == (self.n, self.n) )
    #     # print(R[1:10, 1:10])
    #     # self.assertTrue( is_pos_def(R) ) # test if the covariance operator is positive-definite.
    #     # print(np.linalg.eigvals(R))
    #     self.assertTrue(is_symmetric(R))
    #     self.assertTrue(is_pos_def(R + 0.0001*np.identity(self.n)))

    # def test_TrainOKGT(self):
    #     kernelName = 'Gaussian'
    #     kernelParam = {'sigma':0.5}
    #     okgt = OKGTReg(self.x, self.y, [kernelName], [kernelName], [kernelParam], [kernelParam])
    #     print okgt.r2
    #     plt.scatter(self.x, okgt.f)

    # def test_RowStackUandLambda(self):
    #     # Simulate data from a multi-dimensional model
    #     y, x = SimData_Wang04(self.n)
    #     l = x.shape[1]
    #
    #     # Construct kernels
    #     kernFn = ks.KernelSelector('Gaussian', sigma=0.5)
    #     kernelList = [kernFn]*l
    #
    #     # RowStackUandLambda is a member method, instantiate the object first
    #     okgt_obj = OKGTReg(x, y, ['Gaussian']*l, ['Gaussian'], [{'sigma':0.5}]*l, [{'sigma':0.5}])
    #     UStack, LambdaStack =  okgt_obj.RowStackUandLambda(x, kernelList, xGroup=[[i+1] for i in range(l)])
    #
    #     print 'U row stack =\n', UStack, '\n'
    #     print 'U dimension = ', UStack.shape, '\n'
    #
    #     print 'Lambda stacked list =\n', LambdaStack, '\n'
    #     print 'Length of Lambda stacked list =\n', len(LambdaStack), '\n'
    #     print LambdaStack.shape

    def test_TrainOKGT_ICD(self):
        # Simulate data from a multi-dimensional model
        y, x = SimData_Wang04(self.n)
        l = x.shape[1]

        # Construct kernels
        kernFn = ks.KernelSelector('Gaussian', sigma=0.5)
        # kernelList = [kernFn]*l
        xGroup=[[i+1] for i in range(l)]

        # RowStackUandLambda is a member method, instantiate the object first
        okgt_obj = OKGTReg(x, y, ['Gaussian']*l, ['Gaussian'], [{'sigma':0.5}]*l, [{'sigma':0.5}])
        zeta, r2 = okgt_obj.TrainOKGT_ICD(y, x, [kernFn], [kernFn]*l, xGroup=xGroup)
        # print zeta, r2
        print r2, '\n'

        # Use the old OKGT training method
        g,f,r2  = okgt_obj.TrainOKGT(y, x, [kernFn], [kernFn]*l, xGroup=xGroup)
        print r2