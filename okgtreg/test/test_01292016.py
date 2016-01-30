__author__ = 'panc'

'''
test the new OKGT training algorithm using additive kernel
'''

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import spatial

from okgtreg import *

np.random.seed(25)
data, group = DataSimulator.SimData_Wang04WithInteraction(500)
kernel = Kernel('gaussian', sigma=0.5)

# Original implementation
print "=== Original method ==="
okgt = OKGTReg(data, kernel=kernel, group=group)
start = time.time()
fit = okgt.train()
elapse = time.time() - start
print "R2 = ", fit['r2']
print "time used: ", elapse

# New implementation
print "=== New method ==="
okgt2 = OKGTReg2(data, kernel=kernel, group=group)
start = time.time()
fit2 = okgt2.train()
elapse = time.time() - start
print "R2 = ", fit2['r2']
print "time used: ", elapse

# Difference between g
print "Cosine similarity of g: ", 1 - spatial.distance.cosine(fit['g'], fit2['g'])
print "L2 norm of g1 - g2: ", np.linalg.norm(fit['g'] - fit2['g'])
print "Sup norm of g1 - g2: ", np.abs(fit['g'] - fit2['g']).max()

# plt.scatter(data.y, fit['g'])
