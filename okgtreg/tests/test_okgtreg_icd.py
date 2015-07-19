__author__ = 'panc'

from okgtreg.okgtreg import *
from okgtreg.simulate import *
import okgtreg.kernel_selector as ks

import unittest
import numpy as np
import scipy.spatial.distance as distance
import scipy as sp
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

# Write lines longer than 80 chars in output file
# http://stackoverflow.com/questions/4286544/write-lines-longer-than-80-chars-in-output-file-python
np.set_printoptions(linewidth=300)

class TestOKGTReg(unittest.TestCase):
    def setUp(self):
        self.n = 500
        np.random.seed(10)
        self.y, self.x = SimData_Wang04(self.n)
        self.p = self.x.shape[1]

    def test_TrainOKGT_ICD(self):
        kernelName = ['Laplace']
        kernelParam = [{'alpha':0.5}]

        # kernelName = ['Gaussian']
        # kernelParam = [{'sigma':0.5}]

        xKernelNames = kernelName * self.p
        yKernelName = kernelName
        xKernelParams = kernelParam * self.p
        yKernelParam = kernelParam

        xGroup = [[i+1] for i in range(self.p)]

        okgt_icd = OKGTReg_ICD(self.x, self.y, xKernelNames, yKernelName, xKernelParams, yKernelParam, xGroup=xGroup)
        okgt_icd.TrainOKGT_ICD()

        plt.figure()
        plt.scatter(okgt_icd.y, okgt_icd.g)
        plt.show()

        # print okgt_icd.g
        # print okgt_icd.f
