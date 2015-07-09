__author__ = 'panc'

"""
Create synthetic data

Reference:
[1] "DR for Supervised Learning with RKHSs" Fukumizu 2004
[2] "Estimating Optimal Transformations for Multiple Regression and Correlation" Brieman 1986
"""

import numpy as np
from scipy.special import cbrt # which can takes cubic root of negative values

def SimData_Breiman1(n, sigma=1):
    """ modelName: Breiman1 """
    # y = exp(x^3 + \epsilon)
    # \epsilon ~ N(0,1)
    # x^3 ~ N(0,1)
    epsilon = sigma * np.random.randn(n)
    x3 = np.random.randn(n)
    y = np.exp(x3 + epsilon)
    x = cbrt(x3)
    # Standard data structure is a matrix
    x = np.matrix(x).T
    y = np.matrix(y).T
    return (y, x)