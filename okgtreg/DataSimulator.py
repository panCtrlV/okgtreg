import numpy as np
from scipy.special import cbrt

from okgtreg.Data import Data
from okgtreg.Group import Group

"""
Create synthetic data

Reference:
[1] "DR for Supervised Learning with RKHSs" Fukumizu 2004
[2] "Estimating Optimal Transformations for Multiple Regression and Correlation" Brieman 1986
"""

class DataSimulator(object):
    @staticmethod
    def SimData_Breiman1(n, sigma=1):
        """
        Breiman 1 model
        This is a 1-d model.

            y = exp(x^3 + \epsilon)
            \epsilon ~ N(0,1)
            x^3 ~ N(0,1)

        :param n: sample size
        :param sigma:
        :return:
        """
        epsilon = sigma * np.random.randn(n)
        x3 = np.random.randn(n)
        y = np.exp(x3 + epsilon)
        x = cbrt(x3)
        # return y, x
        return Data(y, x)

    @staticmethod
    def SimData_MultiplyNoise(n):
        """
        Y = X1 + X2 * eps
        :param n:
        :return:
        """
        x = np.random.normal(0, 1, 2*n).reshape((n,2))
        noise = np.random.standard_normal(n)
        y = x[:,0] + x[:,1] * noise
        # return y, x
        return Data(y, x)

    @staticmethod
    def SimData_Wang04(n):
        """
        Model Name: Wang04

            y=log(4 + sin(4 * X1) + |X2| + X3^2 + X4^3 + X5 + 0.1*\epsilon)
            Xi ~ Unif(-1, 1)
            \epsilon ~ N(0, 1)

        This model can be found from: http://partofthething.com/ace/samples.html

        :param n:
        :return:
        """
        # _x = [np.array([np.random.random() * 2.0 - 1.0 for i in range(n)]) for _i in range(0, 5)]
        x = np.vstack(np.random.random(n) * 2.0 - 1.0 for j in range(0, 5)).T
        noise = np.random.standard_normal(n)
        y = np.log(4.0 + np.sin(4 * x[:, 0]) + np.abs(x[:, 1]) + x[:, 2]**2 +
                    x[:, 3]**3 + x[:, 4] + 0.1 * noise)
        # return y, x
        return Data(y, x)

    @staticmethod
    def SimData_Wang04WithInteraction(n):
        """
        Model Name: Wang04 With Interaction

            y=log(4 + sin(4 * X1) + |X2| + X3^2 + X4^3 + X5 + X6*X7 + 0.1*\epsilon)
            Xi ~ Unif(-1, 1)
            \epsilon ~ N(0, 1)

        :param n:
        :return:
        """
        group = Group([1], [2], [3], [4], [5], [6,7])

        # x = np.vstack(np.random.random(n) * 2.0 - 1.0 for j in range(7)).T
        x = np.random.uniform(-1., 1., (n, group.p))
        noise = np.random.standard_normal(n) * 0.1
        h = 4.0 + \
            np.sin(4 * x[:, 0]) + \
            np.abs(x[:, 1]) + \
            x[:, 2] ** 2 + \
            x[:, 3] ** 3 + \
            x[:, 4] + \
            x[:, 5] * x[:, 6] + \
            noise
        y = np.log(h)

        return Data(y, x), group, h

    @staticmethod
    def SimData_Wang04WithInteraction2(n):
        """
        Modification of Wang04WithInteraction where the interaction is 3-way:

            y=log(4 + sin(4 * X1) + |X2| + X3^2 + X4^3 + X5 + X6*X7*X8 0.1*\epsilon)
            Xi ~ Unif(-1, 1)
            \epsilon ~ N(0, 1)

        :param n:
        :return:
        """
        x = np.random.uniform(-1., 1., (n, 8))
        noise = np.random.standard_normal(n)
        y = np.log(4.0 +
                   np.sin(4 * x[:, 0]) +
                   np.abs(x[:, 1]) +
                   x[:, 2]**2 +
                   x[:, 3]**3 +
                   x[:, 4] +
                   abs(x[:, 5] * x[:, 6] * x[:, 7]) +
                   0.1 * noise)
        return Data(y, x)

    @staticmethod
    def SimData_Wang04WithTwoBivariateGroups(n):
        """
        Model Name: Wang04 With Two Bivariate Groups

            y = (1 + sin(2 * X1) + |X2| + X3^2 + X4^3 + X5 + X6*X7 + 0.1 * \epsilon)^(1/3)
            Xi ~ Unif(-1, 1) -> Xi ~ Unif(-pi, pi)
            \epsilon ~ N(0, 1)

        :type n: int
        :param n: sample size

        :rtype: tuple(Data, Group)
        :return: data and the true group structure
        """
        p = 9
        # x = (np.random.random(n*p) * 2.0 - 1.0).reshape((n, p))
        x = np.random.uniform(-np.pi, np.pi, n*p).reshape((n, p))
        noise = np.random.standard_normal(n) * 0.1
        gy = 1. + np.sin(2. * x[:, 0]) + \
             np.abs(x[:, 1]) + \
             x[:, 2]**2 + \
             x[:, 3]**3 + \
             x[:, 4] + \
             x[:, 5] * x[:, 6] + \
             np.cos(x[:, 7] + x[:, 8]) + \
             noise
        y = np.sign(gy) * np.power(np.abs(gy), 1./3)

        group = Group([1], [2], [3], [4], [5], [6,7], [8,9])

        return Data(y, x), group

    @staticmethod
    def SimData_Wang04WithInteraction2_100(n):
        """
        Modification of Wang04WithInteraction where the interaction is 3-way:

            y=log(4 + sin(4 * X1) + |X2| + X3^2 + X4^3 + X5 + X6*X7*X8 0.1*\epsilon)
            Xi ~ Unif(-1, 1)
            \epsilon ~ N(0, 1)

        :param n:
        :return:
        """
        x = np.random.uniform(-1., 1., (n, 8))
        noise = np.random.standard_normal(n)
        y = np.log(4.0 +
                   np.sin(4 * x[:, 0]) +
                   np.abs(x[:, 1]) +
                   x[:, 2]**2 +
                   x[:, 3]**3 +
                   x[:, 4] +
                   100. * abs(x[:, 5] * x[:, 6] * x[:, 7]) +
                   0.1 * noise)
        return Data(y, x)
