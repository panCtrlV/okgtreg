__author__ = 'panc'

'''
This is another simple model. It is different from that
in the simulation study "sim_01172016" in that the current
model uses more complex transformations, such as sin and
absolute functions.
'''

import numpy as np
from scipy.special import expit
from okgtreg import *


def simpleData(n):
    '''
    An simple model whose data is used in the current
    simulation. The model is:

        y = ( sin(x_1) + log(x_2 + x_3) + sigmoid(x_1 + x_2 + x_3) + \epsilon )^2

        x_i \sim Unif(-1, 1)
        \epsilon \sim N(0, 0.1^2)

    :param n:
    :return:
    '''
    x = np.random.uniform(-1., 1., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = (np.sin(x[:, 0]) + np.log(x[:, 1] * x[:, 2]) +
         expit(x[:, 3] * x[:, 4] * x[:, 5]) + e) ** 2
    return Data(y, x), Group([1], [2, 3], [4, 5, 6])
