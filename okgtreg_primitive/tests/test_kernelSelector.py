__author__ = 'panc'

#from unittest import TestCase
import unittest
import okgtreg_primitive.kernel_selector as ks

class TestKernelSelector(unittest.TestCase):
    def testGaussian(self):
        gaussKernel = ks.KernelSelector('Gaussian', sigma=0.5)
        x = (-10, 0, 10, 1, -1)
        y = (10.5, 0, 9.5, -2, 3)
        return [gaussKernel(x[i], y[i]) for i in range(5)]

    def testExpo(self):
        expoKernel = ks.KernelSelector('Exponential', sigma=0.5)
        x = (-10, 0, 10, 1, -1)
        y = (10.5, 0, 9.5, -2, 3)
        return [expoKernel(x[i], y[i]) for i in range(5)]

    def testLaplace(self):
        laplaceKernel = ks.KernelSelector('Laplace', alpha=0.5)
        x = (-10, 0, 10, 1, -1)
        y = (10.5, 0, 9.5, -2, 3)
        return [laplaceKernel(x[i], y[i]) for i in range(5)]

    def testPolynomial(self):
        polyKernel = ks.KernelSelector('Polynomial', slope=1, intercept=2, degree=3)
        x = (-10, 0, 10, 1, -1)
        y = (10.5, 0, 9.5, -2, 3)
        return [polyKernel(x[i], y[i]) for i in range(5)]

    def testSigmoid(self):
        sigmoidKernel = ks.KernelSelector('Sigmoid', a=3, r=1)
        x = (-10, 0, 10, 1, -1)
        y = (10.5, 0, 9.5, -2, 3)
        return [sigmoidKernel(x[i], y[i]) for i in range(5)]

if __name__=='__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKernelSelector)
    unittest.TextTestRunner(verbosity=2).run(suite)