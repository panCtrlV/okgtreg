__author__ = 'panc'

import numpy as np
from okgtreg.Group import Group
from okgtreg.Data import Data

p = 6


# model 1
# balanced group structure
# h = \sqrt{|x1*x2|} + \sin((x3 + x4)*pi) + \cos((x5 + x6)*pi)
# y = 1 / (1 + h)
def model1(n):
    x = np.random.standard_normal((n, p)) * np.sqrt(3)
    e = np.random.standard_normal((n,)) * 0.1
    g = np.sqrt(np.abs(x[:, 0] * x[:, 1])) + \
        np.sin(np.pi * x[:, 2:4].sum(axis=1)) + \
        np.cos(np.pi * x[:, 2:].sum(axis=1)) + \
        e
    y = 1. / (1. + g ** 2)
    return Data(y, x), Group([1, 2], [3, 4], [5, 6]), g


# model 2
# unbalanced group structure
# h = 1/(1+x1^2) + arcsin((x2+x3) / 2) + arctan((x4+x5+x6)^3)
# y = h^2
def model2(n):
    x = np.random.uniform(-1, 1, (n, p))
    e = np.random.standard_normal((n,)) * 0.01
    g = 1. / (1 + x[:, 0] ** 2) + \
        np.arcsin(x[:, 1:3].sum(axis=1) / 2) + \
        np.arctan(x[:, 3:].sum(axis=1) ** 3) + \
        e
    y = g ** 2
    return Data(y, x), Group([1], [2, 3], [4, 5, 6]), g


def selectModel(id):
    models = {1: model1,
              2: model2}
    return models[id]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(25)
    data, group, h = model1(500)
    print data
    plt.hist(data.y, 30)

    np.random.seed(25)
    data, group, h = model2(500)
    print data
    plt.hist(data.y, 30)
