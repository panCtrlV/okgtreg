__author__ = 'panc'

import numpy as np

from okgtreg.Data import Data
from okgtreg.Group import Group

p = 6  # number of covariates


def model1(n):
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,)) * 0.1
    g = 2. * x[:, 0] + \
        x[:, 1] ** 2 + \
        x[:, 2] ** 3 + \
        np.sin(x[:, 3]) + \
        np.log(x[:, 4] + 5.) + \
        np.abs(x[:, 5]) + \
        e
    y = g ** 2
    return Data(y, x), Group([1], [2], [3], [4], [5], [6])


def model2(n):
    x = np.random.standard_normal((n, p - 2))
    x2 = np.random.uniform(1, 3, (n, 2))
    e = np.random.standard_normal((n,)) * 0.1
    g = (x[:, :2].sum(axis=1)) ** 2 + \
        np.log((x[:, 2:4] ** 2).sum(axis=1) + 5.) + \
        x2[:, 0] ** x2[:, 1] + \
        e
    y = g ** 2
    return Data(y, x), Group([1, 2], [3, 4], [5, 6])


def model3(n):
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,)) * 0.1
    g = (x[:, :3].sum(axis=1)) ** 2 + \
        np.log((x[:, 3:] ** 2).sum(axis=1) + 5.) + \
        e
    y = g ** 2
    return Data(y, x), Group([1, 2, 3], [4, 5, 6])


def model4(n):
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,))
    g = np.exp(np.linalg.norm(x, ord=2, axis=1)) + e
    y = np.sqrt(g)
    return Data(y, x), Group([1, 2, 3, 4, 5, 6])


def selectModel(id):
    models = {1: model1,
              2: model2,
              3: model3,
              4: model4}
    return models[id]


if __name__ == '__main__':
    data, group = selectModel(1)(500)
    print data
    print group

    data, group = selectModel(2)(500)
    print data
    print group

    data, group = selectModel(3)(500)
    print data
    print group

    data, group = selectModel(4)(500)
    print data
    print group
