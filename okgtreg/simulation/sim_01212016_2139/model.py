__author__ = 'panc'

import numpy as np

from okgtreg.Data import Data
from okgtreg.Group import Group


def simplePolyModel(n):
    x = np.random.normal(0., 1., (n, 6))
    e = np.random.normal(0., 0.1, n)
    y = np.log((x[:, :3].sum(1)) ** 2 + (x[:, 3:].sum(1)) ** 2 + e)
    return Data(y, x), Group([1, 2, 3], [4, 5, 6])
