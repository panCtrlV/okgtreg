__author__ = 'panc'

import numpy as np

'''
Rank group structure after imposing penalty
for each pair of (mu, a)
'''

# Tuning parameter \mu
# mu = 1e-4
mu = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), 10))
a = np.arange(1, 11)
