__author__ = 'panc'

import numpy as np
from scipy.special import expit, cbrt

from okgtreg import *


def model2(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = np.log((1. + x.dot(b) + e) ** 2)
    return Data(y, x), Group([1], [2], [3], [4], [5], [6])


def model5(n):
    # TODO: fix coefficients
    b = np.random.normal(0., 1., 6)
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = cbrt(1. +
             x[:, :3].dot(b[:3]) +
             np.exp(x[:, 3:].dot(b[3:])) +
             e)
    return Data(y, x), Group([1], [2], [3], [4, 5, 6])


def model6(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.normal(0., 1., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = np.log(np.abs(1. +
                      x[:, :3].dot(b[:3]) +
                      np.log(np.abs(x[:, 3:].dot(b[3:]))) +
                      e))
    return Data(y, x), Group([1], [2], [3], [4, 5, 6])


def model7(n):
    b = np.random.normal(0., 1., 6)
    x = np.random.uniform(0., 2., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = expit(1. +
              x[:, :3].dot(b[:3]) +
              np.abs(x[:, 3:].dot(b[3:])) +
              e)
    return Data(y, x), Group([1], [2], [3], [4, 5, 6])


def selectModel(id):
    models = {2: model2,
              5: model5,
              6: model6,
              7: model7}
    return models[id]
