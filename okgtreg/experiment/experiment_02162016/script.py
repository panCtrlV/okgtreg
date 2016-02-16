__author__ = 'panc'

'''
In OKGT, if the response transformation is assumed to be known.
Then, it is conjectured that the resulting OKGT fitting is equivalent
to a LS regression where the design matrix is the Gram matrix of a
additive kernel.
'''

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from okgtreg.Data import ParameterizedData
from okgtreg.Parameters import Parameters
from okgtreg.Kernel import Kernel
from okgtreg.DataSimulator import DataSimulator
from okgtreg.OKGTReg import OKGTReg2

# Simulate data
np.random.seed(25)
data, true_group_struct, h = DataSimulator.SimData_Wang04WithInteraction(500)

# Kernel for one group
kernel = Kernel('gaussian', sigma=0.5)
parameters = Parameters(true_group_struct, ykernel=kernel, xkernels=[kernel] * true_group_struct.size)
PData = ParameterizedData(data, parameters)
Kx_list = PData._getGramsForX(centered=False)
K_add = sum(Kx_list)

#######################
# LS regression h ~ K #
#######################
# We use the Python provided function for LS regression
# Reference: multivariate linear regression in Python
# http://stackoverflow.com/questions/11479064/multivariate-linear-regression-in-python
clf = linear_model.LinearRegression()
clf.fit(K_add, h)
beta = clf.coef_
b0 = clf.intercept_
# Check the covariate transformations
## Covariate transformations
f_list = [Kx_list[j].dot(beta) for j in range(true_group_struct.size)]
h_pred = sum(f_list) + b0
r2 = 1 - sum((h - h_pred) ** 2) / sum((h - np.mean(h)) ** 2)
print "R2 from LS regression: %.05f" % r2

## 1-d transformations
fig, axarr = plt.subplots(2, 3)
row_idx = -1
for i in range(len(f_list) - 1):
    if i % 3 == 0:
        row_idx += 1
    axarr[row_idx, i % 3].set_title(r'$f_%d$' % (i + 1))
    axarr[row_idx, i % 3].scatter(data.X[:, i], f_list[i], s=0.5)
# j=4; plt.scatter(data.X[:,j], f_list[j])

## 2-d transformation
## Reference: 3d plot in Python
## http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#scatter-plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 5], data.X[:, 6], f_list[5])

# Alternatively, we can run the regression as follows.
# We augment an ones-vector in front of K_add,and use
# the augmented matrix, K_add_aug, as the design matrix.
# The fixed/known response transformation h is the response.
#
# We can analyze the linear regression h ~ K_add_aug as
# follows.
#
# The application of SVD gives
#
#   K_add_aug = U * S * V
#
# Then, after some derivation, we get
#
#   beta_hat = V^T * S^{-1} * U^T * h
K_add_aug = np.hstack([np.ones(500)[:, np.newaxis], K_add])
U, s, V = np.linalg.svd(K_add_aug, full_matrices=False)
# thresholding the singular values
mask = (s > 1e-10)
s_inv_masked = np.concatenate((1. / s[mask], np.zeros(500 - sum(mask))))
beta2 = reduce(np.dot, [V.T, np.diag(s_inv_masked), U.T, h])
f_list2 = [Kx_list[j].dot(beta2[1:]) for j in range(true_group_struct.size)]
# prediction
h_pred2 = sum(f_list2) + beta2[0]
r22 = 1 - sum((h - h_pred2) ** 2) / sum((h - np.mean(h)) ** 2)
print "R2 from LS regression: %.05f" % r22
# plot transformations
fig, axarr = plt.subplots(2, 3)
row_idx = -1
for i in range(len(f_list2) - 1):
    if i % 3 == 0:
        row_idx += 1
    axarr[row_idx, i % 3].set_title(r'$f_%d$' % (i + 1))
    axarr[row_idx, i % 3].scatter(data.X[:, i], f_list2[i], s=0.5)

#################################
# What if using OKGT with fixed #
# response transformation       #
#################################
okgt = OKGTReg2(data, kernel=kernel, group=true_group_struct)
fit = okgt._train_Vanilla2(h)
print "R2 from fitting OKGT: %.05f" % fit['r2']

# Check covariate transformations
## 1-d functions
fig, axarr = plt.subplots(2, 3)
row_idx = -1
for i in range(fit['f'].shape[1] - 1):
    if i % 3 == 0:
        row_idx += 1
    axarr[row_idx, i % 3].set_title(r'$f_%d$' % (i + 1))
    axarr[row_idx, i % 3].scatter(data.X[:, i], fit['f'][:, i], s=0.5)


    # j=4; plt.scatter(data.X[:,j], fit['f'][:,j])
