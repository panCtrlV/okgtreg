__author__ = 'panc'

import unittest
from okgtreg.mymath import *
from scipy import random, linalg
import time

class TestMyMath(unittest.TestCase):
    def setUp(self):
        self.matrixSize = 10000
        A = random.rand(self.matrixSize, self.matrixSize)
        self.M = np.matrix(np.dot(A,A.transpose()))

    def testMatrixInverse_correctness(self):
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

    def testMatrixInverse_time(self):
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