__author__ = 'panc'

'''
This is another simple model. It is different from that
in the simulation study "sim_01172016" in that the current
model uses more complex transformations, such as sin and
absolute functions.
'''

import numpy as np
from scipy.special import expit, cbrt
from okgtreg import *


def simpleData_01192016(n):
    '''
    An simple model whose data is used in the current
    simulation. The model is:

        y = ( sin(x_1) + log(|x_2 + x_3|) + sigmoid(x_1 + x_2 + x_3) + \epsilon )^(1/3)

        x_i \sim Unif(-1, 1)
        \epsilon \sim N(0, 0.1^2)

    :param n:
    :return:
    '''
    x = np.random.normal(0., 1., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = cbrt(np.sin(x[:, 0]) +
             np.log(np.abs(x[:, 1] * x[:, 2])) +
             expit(x[:, 3] * x[:, 4] * x[:, 5]) +
             e)
    return Data(y, x), Group([1], [2, 3], [4, 5, 6])


if __name__ == '__main__':
    np.random.seed(25)
    n = 500
    data, group = simpleData_01192016(n)

    import matplotlib.pyplot as plt

    plt.scatter(np.arange(n), data.y)  # scatter plot
    plt.hist(data.y, 30)  # histogram shows bi-modal
