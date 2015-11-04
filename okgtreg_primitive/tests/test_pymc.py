__author__ = 'panc'

import unittest
import pymc.gp.incomplete_chol as pyichol
import numpy as np

class TestPyMC(unittest.TestCase):
    def setUp(self):
        self.n = 500
        x = np.matrix(np.random.randn(self.n ** 2).reshape((self.n, self.n)))
        self.x = x * x.T # ps matrix

    def test_ichol_full(self):
        """
        Test if the factorization is close to the original matrix.
        """
        mrange = np.arange(self.n) + 1

        # def rowfun(i,x,rowvec): return
        # x_diag = np.diag(self.x) # diagonal of the square matrix
        reltol = 1e-6

        for i in mrange:
            pass
            # TODO: to be completed