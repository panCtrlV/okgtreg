__author__ = 'panc'

import numpy as np
from scipy.special import expit, cbrt

from okgtreg import *


def model1(n):
    b = np.random.normal(0., 1., 6)
    # x = np.random.normal(0., 2., (n, 6))
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = (1. + x.dot(b) + e) * 3
    return Data(y, x), Group([1], [2], [3], [4], [5], [6])


def model2(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = np.log((1. + x.dot(b) + e) ** 2)
    return Data(y, x), Group([1], [2], [3], [4], [5], [6])


def model3(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = (1. + x[:, :3].dot(b[:3]) + 10. * np.sin(x[:, 3:].dot(b[3:])) + e) ** 2
    return Data(y, x), Group([1], [2], [3], [4, 5, 6])


def model4(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = (1. + x[:, :3].dot(b[:3]) + 10. * expit(x[:, 3:].dot(b[3:])) + e) ** 2
    return Data(y, x), Group([1], [2], [3], [4, 5, 6])


def model5(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = cbrt(1. + x[:, :3].dot(b[:3]) + np.exp(x[:, 3:].dot(b[3:])) + e)
    return Data(y, x), Group([1], [2], [3], [4, 5, 6])


def model6(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.standard_t(1, (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = np.log(np.abs(1. + x[:, :3].dot(b[:3]) + np.log(np.abs(x[:, 3:].dot(b[3:]))) + e))
    return Data(y, x), Group([1], [2], [3], [4, 5, 6])


def model7(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = expit(1. + x[:, :3].dot(b[:3]) + np.abs(x[:, 3:].dot(b[3:])) + e)
    return Data(y, x), Group([1], [2], [3], [4, 5, 6])


def model8(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.normal(1., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = cbrt(1. + \
             b[0] * x[:, 0] + \
             b[1] * x[:, 1] ** 2 + \
             b[2] * x[:, 2] ** 3 + \
             b[3] * np.abs(x[:, 3]) + \
             b[4] * np.sin(x[:, 4]) + \
             b[5] * np.exp(x[:, 5]) + e)
    return Data(y, x), Group([1], [2], [3], [4], [5], [6])


def model9(n):
    # x = np.random.normal(0., 2., (n, 6))
    x = np.random.uniform(1, 2, (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = np.log(np.abs(1. +
                      np.sin(x[:, 0]) +
                      x[:, 1] * x[:, 2] +
                      expit(x[:, 3] * x[:, 4] * x[:, 5]) + e))
    return Data(y, x), Group([1], [2, 3], [4, 5, 6])


def model10(n):
    x = np.random.uniform(1, 2, (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = np.log(1. +
               np.log(x[:, 0]) +
               x[:, 1] / np.exp(x[:, 2]) +
               np.power(x[:, 3] + x[:, 4], x[:, 5]) + e)
    return Data(y, x), Group([1], [2, 3], [4, 5, 6])


def selectModel(id):
    models = {1: model1,
              2: model2,
              3: model3,
              4: model4,
              5: model5,
              6: model6,
              7: model7,
              8: model8,
              9: model9,
              10: model10}
    return models[id]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data, group = model1(500)
    plt.hist(data.y, 30)

    data, group = model2(500)
    plt.hist(data.y, 30)

    data, group = model3(500)
    plt.hist(data.y, 30)
    plt.boxplot(data.y)

    data, group = model4(500)
    plt.hist(data.y, 30)

    data, group = model5(500)
    plt.hist(data.y, 30)

    data, group = model6(500)
    plt.hist(data.y, 30)

    data, group = model7(500)
    plt.hist(data.y, 30)

    data, group = model8(500)
    plt.hist(data.y, 30)

    data, group = model9(500)
    plt.hist(data.y, 30)

    data, group = model10(500)
    plt.hist(data.y, 30)

    model = selectModel(1)
    data, group = model(500)
