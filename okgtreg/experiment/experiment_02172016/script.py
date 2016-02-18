__author__ = 'panc'

'''
After talking with Qiming, I would like to perform the following
experiment.

In this experiment, I would like to compare the stacked approach and
additive approach of OKGT fitting on the following synthesized model:

    There are two covariates X1 and X2, while the model only
    depends on the first covariate as follows:

        h = sin(x1) + e

    So if we want to explicitly include X2 in the mode, it is
    written as:

        h = sin(x1) + f(x2) + e
        f(x2) = 0 for all x2

I want to see if the two approaches of OKGT fitting can recover the
true transformations.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from okgtreg.Data import Data
from okgtreg.Group import Group
from okgtreg.Kernel import Kernel
from okgtreg.OKGTReg import OKGTReg, OKGTReg2


# Create model
def model(n):
    p = 2
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal(n) * 0.1
    h = np.sin(x[:, 0] * np.pi) + e
    y = h ** 3
    return Data(y, x), Group([1], [2]), h


# Simulate data
n = 500
np.random.seed(25)
data, true_gstruct, h = model(n)

# plt.hist(data.y, 30)

# Kernel
# kernel = Kernel('gaussian', sigma=0.5)
kernel = Kernel('laplace', sigma=0.5)

# OKGTReg using stacked approach
okgt_stack = OKGTReg(data, kernel=kernel, group=true_gstruct)
fit_stack = okgt_stack.train()
print "R2 (stacked) = %.05f" % fit_stack['r2']

# OKGTReg using additive approach
okgt_add = OKGTReg2(data, kernel=kernel, group=true_gstruct)
fit_add = okgt_add.train()
print "R2 (additive) = %.05f" % fit_add['r2']

# Plot transformations
## list all four plots in a figure
fig, axarr = plt.subplots(2, 2)
fig.suptitle("Stacked vs Additive OKGT Fitting " +
             r"$y = (sin(x_1) + \epsilon$)^3")

axarr[0, 0].set_title("Stacked OKGT fitting" + r'$f_1$')
axarr[0, 0].scatter(data.X[:, 0], fit_stack['f'][:, 0], s=0.5)
axarr[0, 1].set_title("Stacked OKGT fitting" + r'$f_2$')
axarr[0, 1].scatter(data.X[:, 0], fit_stack['f'][:, 1], s=0.5)

axarr[1, 0].set_title("Additive OKGT fitting" + r'$f_1$')
axarr[1, 0].scatter(data.X[:, 0], fit_add['f'][:, 0], s=0.5)
axarr[1, 1].set_title("Additive OKGT fitting" + r'$f_2$')
axarr[1, 1].scatter(data.X[:, 0], fit_add['f'][:, 1], s=0.5)

# The following inspects their coefficients (we need to
# modify the codes of OKGTReg._train_vanilla and
# OKGTReg2._train_vanilla to capture the coefficients.)
from scipy.spatial.distance import cosine

## coefficients from two approaches
alpha_stack = fit_stack['coef']
alpha_add = fit_add['coef']
## calculate cosine distances
cosine(alpha_stack[:, 0], alpha_add.squeeze())
cosine(alpha_stack[:, 1], alpha_add.squeeze())
cosine(alpha_stack[:, 0], alpha_stack[:, 1])
## check if transformations are similar
Kxc_list = okgt_stack.parameterizedData._getGramsForX()
K1c = Kxc_list[0]
K2c = Kxc_list[1]

plt.scatter(data.X[:, 0], K1c.dot(alpha_stack[:, 0]))
plt.scatter(data.X[:, 1], K2c.dot(alpha_stack[:, 1]))

plt.scatter(data.X[:, 0], K1c.dot(alpha_add), color='red')
plt.scatter(data.X[:, 1], K2c.dot(alpha_add), color='red')
## check average of the coefficients from stacked approach
## is similar to the coefficients from additive approach
cosine(([0.5, 0.5] * alpha_stack).sum(axis=1), alpha_add.squeeze())
cosine(np.hstack([alpha_stack[:, 0], alpha_stack[:, 1]]),
       np.hstack([alpha_add.squeeze(), alpha_add.squeeze()]))

# What if we use a bi-variate kernel function?
#
# It is conjectured that the dimension of X2 would
# vanish since the model is independent of the covariate.
gstruct2 = Group([1, 2])

okgt2_stack = OKGTReg(data, kernel=kernel, group=gstruct2)
fit2_stack = okgt2_stack.train()
print "R2 (stacked, bi-variate kernel) = %.05f" % fit2_stack['r2']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 0], data.X[:, 1], fit2_stack['f'][:, 0], s=0.5)

okgt2_add = OKGTReg2(data, kernel=kernel, group=gstruct2)
fit2_add = okgt2_add.train()
print "R2 (additive, bi-variate kernel) = %.05f" % fit2_add['r2']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 0], data.X[:, 1], fit2_add['f'][:, 0], s=0.5)
