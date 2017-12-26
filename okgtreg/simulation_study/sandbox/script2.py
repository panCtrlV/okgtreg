# -*- coding: utf-8 -*-
# @Author: Pan Chao
# @Date:   2017-05-12 23:57:19
# @Last Modified by:   Pan Chao
# @Last Modified time: 2017-12-26 10:46:49


'''
Check subspace Theorem
If the difference of two gram matrices is positive definite
'''


import numpy as np
import pickle
import os, sys
from okgtreg.Kernel import Kernel


args = sys.argv
sd = args[1]  # standard deviation


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# Use Gaussian kernel
kernel = Kernel('gaussian', sigma=0.5)

# generate data
n = 500
p = 2

# sd = 100

res = []
counter = 0
while counter < 100:
    counter += 1
    x = np.random.normal(0, 100, (n, p))

    # One is 2-d gaussian
    G2 = kernel.gram(x, centered=False)
    # The second is the sum of two gram matrices
    G11 = kernel.gram(x[:, 0][:, np.newaxis], centered=False) + \
          kernel.gram(x[:, 1][:, np.newaxis], centered=False)

    # Difference
    B = 10
    D = B ** 2 * G2 - G11

    # print np.linalg.eigvals(D)
    ispos = is_pos_def(D)
    print counter, " : ", ispos
    res.append(ispos)

cwd = os.getcwd()
fname = "pd_test_" + sd + '.pkl'
saveto = cwd + '/' + fname
with open(saveto, 'wb') as f:
    pickle.dump(res, f)
