__author__ = 'panc'

'''
test the new OKGT training algorithm using additive kernel
'''

import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import spatial

from okgtreg import *

n = 1000
np.random.seed(25)
data, group = DataSimulator.SimData_Wang04WithInteraction(n)
kernel = Kernel('gaussian', sigma=0.5)

print "# ==="
print "# Sample size: ", n
print "# Kernel: Gaussian (0.5)"
print "# ===\n"

# Original implementation
print "=== Original method ==="
okgt = OKGTReg(data, kernel=kernel, group=group)
start = time.time()
fit = okgt.train()
elapse = time.time() - start
print "R2 = ", fit['r2']
print "time used: ", elapse, '\n'

# New implementation
print "=== New method ==="
okgt2 = OKGTReg2(data, kernel=kernel, group=group)
start = time.time()
fit2 = okgt2.train()
elapse = time.time() - start
print "R2 = ", fit2['r2']
print "time used: ", elapse, '\n'

# Difference between g
print "=== Difference between the estimates of g ==="
print "Cosine similarity of g: ", 1 - spatial.distance.cosine(fit['g'], fit2['g'])
print "L2 norm of g1 - g2: ", np.linalg.norm(fit['g'] - fit2['g'])
print "Sup norm of g1 - g2: ", np.abs(fit['g'] - fit2['g']).max()

# # Plot two g's
# plt.scatter(data.y, fit['g'])
# plt.scatter(data.y, fit2['g'])
#
# # Plot f
# j=4; plt.scatter(data.X[:,j], fit2['f'][:,j])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data.X[:,5], data.X[:,6], fit2['f'][:,5])
