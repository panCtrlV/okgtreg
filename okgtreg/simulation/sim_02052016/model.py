__author__ = 'panc'

import numpy as np
from okgtreg.Group import Group
from okgtreg.Data import Data

from scipy.special import cbrt

p = 6


# model 1
# balanced group structure
# h = \sqrt{|x1*x2|} + \sin((x3 + x4)*pi) + \cos((x5 + x6)*pi)
# y = 1 / (1 + h)
def model1(n):
    x = np.random.standard_normal((n, p)) * np.sqrt(3)
    e = np.random.standard_normal((n,)) * 0.01
    h = np.sqrt(np.abs(x[:, 0] * x[:, 1])) + \
        np.sin(np.pi * x[:, 2:4].sum(axis=1)) + \
        np.cos(np.pi * x[:, 4:].sum(axis=1)) + \
        e
    y = 1. / (1. + h ** 2)
    return Data(y, x), Group([1, 2], [3, 4], [5, 6]), h

# model 2
# unbalanced group structure
# h = 1/(1+x1^2) + arcsin((x2+x3) / 2) + arctan((x4+x5+x6)^3)
# y = h^2
def model2(n):
    x = np.random.uniform(-1, 1, (n, p))
    e = np.random.standard_normal((n,)) * 0.01
    h = 1. / (1 + x[:, 0] ** 2) + \
        np.arcsin(x[:, 1:3].sum(axis=1) / 2) + \
        np.arctan(x[:, 3:].sum(axis=1) ** 3) + \
        e
    y = h ** 2
    return Data(y, x), Group([1], [2, 3], [4, 5, 6]), h


# model 3
# fully additive
# h = 2*x1 + x2**2 + x3**3 + sin(x4*pi) + log(x5+5) + |x6|
# y = ln(h^2)
def model3(n):
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,)) * 0.01
    h = 2. * x[:, 0] + \
        x[:, 1] ** 2 + \
        x[:, 2] ** 3 + \
        np.sin(x[:, 3] * np.pi) + \
        np.log(np.abs(x[:, 4] + 5.)) + \
        np.abs(x[:, 5]) + \
        e
    y = np.log(h ** 2)
    return Data(y, x), Group([1], [2], [3], [4], [5], [6]), h


# model 4
# Fully non-parametric
# h = exp{(x1^2 + ... + x6^2)^{1/2}}
# y = h^{1/2}
def model4(n):
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,)) * 0.01
    h = np.exp(np.linalg.norm(x, ord=2, axis=1)) + e
    y = np.sqrt(h)
    return Data(y, x), Group([1, 2, 3, 4, 5, 6]), h


# model 5
# Unbalanced group structure, as compared with model 1
# h = arcsin((x1+x3) / 2) + 1/(1+x2^2) + arctan((x4+x5+x6)^3)
# y = h^2
def model5(n):
    x = np.random.uniform(-1, 1, (n, p))
    e = np.random.standard_normal((n,)) * 0.01
    h = 1. / (1 + x[:, 1] ** 2) + \
        np.arcsin(x[:, [0, 2]].sum(axis=1) / 2) + \
        np.arctan(x[:, 3:].sum(axis=1) ** 3) + \
        e
    y = h ** 2
    return Data(y, x), Group([1, 3], [2], [4, 5, 6]), h


# model 6
# Unbalanced two group
# h = arctan((1+x1+x2) / (1+x3+x4)) + (x5 + |x6|^{1/2})^{1/3}
# y = ln(h^4)
def model6(n):
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,)) * 0.01
    h = np.arctan((1 + x[:, 0] + x[:, 1]) / (1 + x[:, 2] + x[:, 3])) + \
        cbrt(x[:, 4] + np.sqrt(np.abs(x[:, 5]))) + \
        e
    y = np.log(h ** 4)
    return Data(y, x), Group([1, 2, 3, 4], [5, 6]), h


def selectModel(id):
    models = {1: model1,
              2: model2,
              3: model3,
              4: model4,
              5: model5,
              6: model6}
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

    np.random.seed(25)
    data, group, h = model3(500)
    print data
    plt.hist(data.y, 30)

    np.random.seed(25)
    data, group, h = model4(500)
    print data
    plt.hist(data.y, 30)

    np.random.seed(25)
    data, group, h = model5(500)
    print data
    plt.hist(data.y, 30)

    np.random.seed(25)
    data, group, h = model6(500)
    print data
    plt.hist(data.y, 30)
