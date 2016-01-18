__author__ = 'panc'

import numpy as np
from okgtreg import *


def simpleData(n):
    '''
    An simple model whose data is used in the current
    simulation. The model is:

        y = ( x_1 + x_2 * x_3 + x_4 * x_5 * x_6 + \epsilon )^2

        x_i \sim Unif(0, 1)
        \epsilon \sim N(0, 0.1^2)

    :param n:
    :return:
    '''
    x = np.random.uniform(0., 1., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = (x[:, 0] + (x[:, 1] * x[:, 2]) + (x[:, 3] * x[:, 4] * x[:, 5]) + e) ** 2
    return Data(y, x), Group([1], [2, 3], [4, 5, 6])
