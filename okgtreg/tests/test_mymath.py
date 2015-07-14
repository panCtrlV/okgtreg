__author__ = 'panc'

import unittest
from okgtreg.mymath import *
from scipy import random, linalg
import time

from okgtreg.okgtreg import *
from okgtreg.simulate import *
import okgtreg.kernel_selector as ks
import okgtreg.mymath as mymath

class TestMyMath(unittest.TestCase):
    def setUp(self):
        self.matrixSize = 1000
        A = random.rand(self.matrixSize, self.matrixSize)
        self.M = np.matrix(np.dot(A,A.transpose()))


    def test_MatrixInverse_correctness(self):
        myInv = MatrixInverse(self.M)
        # print myInv

        # call np.linalg.inv
        npInv = np.linalg.inv(self.M)
        # print '\n'
        # print npInv

        # Compare the inverse from two methods
        # self.assertTrue((myInv == npInv).all())
        # print '\n'
        print np.abs(myInv - npInv).max()

    def test_MatrixInverse_time(self):
        start = time.time()
        myInv = MatrixInverse(self.M)
        end = time.time()
        print '\n'
        # print 'My inverse: ' + str(end - start)
        print end - start

        start = time.time()
        npInv = np.linalg.inv(self.M)
        end = time.time()
        print '\n'
        # print 'Numpy inverse: ' + str(end - start)
        print end - start

        # Numpy's linalg.inv is faster

    def test_ApplyICDonGramMatrix(self):
        # simulate data from a model
        n = 500
        p = 1
        np.random.seed(10)
        y, x = SimData_Breiman1(n)

        # construct gram matrices (uncentered)
        kernFn = ks.KernelSelector('Gaussian', sigma=0.5)
        yGram = OKGTReg.GramMatrix(y, kernFn, centered=False)
        xGram = OKGTReg.GramMatrix(x, kernFn, centered=False)

        U_y, Lambda_y, pind_y = mymath.ApplyICDonGramMatrix(yGram)
        U_x, Lambda_x, pind_y = mymath.ApplyICDonGramMatrix(xGram)

        print 'U_x =\n', U_x, '\n'
        print 'Lambda_x =\n', Lambda_x, '\n'

        print 'U_y =\n', U_y, '\n'
        print 'Lambda_y =\n', Lambda_y, '\n'


