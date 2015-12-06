"""
In this experiment, we experiment a modified group structure detection procedure where
each time the fitting residual is used to fit a new OKGT. The idea is as follows.

Given the fitting of the current OKGT, we choose either to split one of the multi-variate
group or merge a single covariate with another group. For example,

The group structure for the current OKGT is:

    ([1], [2,3], [4]).

By splitting a multi-variate group, [2,3] is split into two univariate groups [2], [3]. In
order to evaluate the contribution / detriment the split causes on the overall fitting of
OKGT,

By merging a covariate with another group, we can combine [1] and [2,3] as [1,2,3].
"""

import numpy as np
import matplotlib.pyplot as plt

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Kernel import Kernel
from okgtreg.Group import Group
from okgtreg.Parameters import Parameters
from okgtreg.OKGTReg import OKGTReg
from okgtreg.Data import Data

"""
First, we fit an OKGT with the fully additive group structure.
"""
# Simulate data
n = 500

np.random.seed(25)
data = DataSimulator.SimData_Wang04WithInteraction2(n)

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Group structure used for OKGT fitting
group = Group(*tuple([[i+1] for i in range(data.p)]))

# Parameters
parameters = Parameters(group, kernel, [kernel]*group.size)

# Construct OKGT and fit
okgt = OKGTReg(data, parameters)
fit = okgt.train()
fit['r2']
resid = fit['g'].reshape((n,)) - fit['f'].sum(1)

# Plot the residual
plt.scatter(np.arange(n), resid)

# Var(X_{6,7,8} | Y) under the current OKGT:
np.var(fit['f'][:,-3:].sum(1))
"""
We calculated the variance of $f_6 + f_7 + f_8$, which can also be
considered as the conditional variance of X6 - X7 altogether given
Y under the fully additive structure.
"""


"""
Then, we change the group structure to the true group structure.
This, of course, will significantly improve the fitting in terms
of R2.
"""
group2 = Group([1], [2], [3], [4], [5], [6,7,8])
parameters2 = Parameters(group2, kernel, [kernel]*group2.size)
okgt2 = OKGTReg(data, parameters2)
fit2 = okgt2.train()
fit2['r2']
resid2 = fit2['g'].reshape((n,)) - fit2['f'].sum(1)
plt.scatter(range(n), resid2)
np.var(fit2['f'][:, -1])
"""
We calculate the variance of $f_{678}$, i.e. the grouped transformation
of X6 - X7, which can also be considered as the conditional variance of
X6 - X7 altogether given Y under the grouped structure.

Though we hope that this number would be higher than that in the previous
calculation, it is actually smaller. This kills our initial hope that an
improvement in the conditional variance would support using a grouped structure.
"""

"""
Since by using covariates directly does not help determining a group structure,
we turn our attention to the residuals of OKGT fitting.

We wounder what would happen if we use f_6 + f_7 + f_8 + \epsilon and
X6 - X8 as a single group to fit an OKGT. That is, f_6 + f_7 + f_8 + \epsilon
as the observations for the response Y and X6 - X8 as the observations for the
only three covariates X. We call this a **marginal OKGT fitting**.

If there is no further gain by merging the three covariates,
the plot of (f_6 + f_7 + f_8) ~ f_{678} should be close to a straight line.
"""
# Fit f_(6,7,8) ~ f_6 + f_7 + f_8
y2 = fit['f'][:, -3:].sum(1) + resid
x2 = data.X[:, -3:]
data2 = Data(y2, x2)

group3 = Group([1,2,3])
parameters3 = Parameters(group3, kernel, [kernel]*group3.size)
okgt3 = OKGTReg(data2, parameters3)
fit3 = okgt3.train()
fit3['r2']

plt.scatter(fit['f'][:, -3:].sum(1), fit3['f'])
"""
The above plot shows some curvature, indicating fitting a grouped,
indicating that fitting X6 - X8 by a grouped transforamtion would
gain more information.
"""

resid3 = (fit3['g'] - fit3['f']).reshape((n,))
plt.scatter(resid, resid3)
"""
The above two lines plot the residuals obtained from the fully additive
OKGT against the residuals from the marginal OKGT fitting. The plot is
more evident that there is still some systemic pattern (quadratic curvature)
in the former residuals which can be extracted.
"""


group4 = Group([2, 3], [1])
parameters4 = Parameters(group4, kernel, [kernel]*group4.size)
okgt4 = OKGTReg(data2, parameters4)
fit4 = okgt4.train()
fit4['r2']

resid4 = fit4['g'].reshape((n,)) - fit4['f'].sum(1)
plt.scatter(resid, resid4)
"""
In the above codes, we break X6 - X8 into two groups, one univariate and the
other bivariate. The redisual-residual plot does not show the improvement of
OKGT fitting.

But, we are considering fitting for a bivariate group. The following experiment
would be more reasonable. That is, if we are grouping X7 and X8 together, we should
just consider y = f_7 + f_8 + \epsilon as the response for the marginal OKGT.
"""

y3 = fit['f'][:, -2:].sum(1) + resid
x3 = data.X[:, -2:]
data3 = Data(y3, x3)
group5 = Group([1,2])
parameters5 = Parameters(group5, kernel, [kernel]*group5.size)
okgt5 = OKGTReg(data3, parameters5)
fit5 = okgt5.train()
fit5['r2']

resid5 = fit5['g'].reshape((n,)) - fit5['f'].sum(1)
plt.scatter(resid, resid5)
"""
The residual-residual plot obtained above shows some non-linear pattern, which indicate
that by grouping X7 and X8 we could gain a better OKGT fitting.
"""

# TODO: Can we use a non-linearity measure as an indicator for the phenomenon shown in
# todo: the above residual-residual plots?

"""
We attempted to use curvature to quantify the linearity of a residual-residual curve.
"""
# Curvature
def curvature(points, returnMean=True):
    """
    Calculate the curvature for a 2d curve.

    :type points: 2d array, size n*2
    :param points: set of points for which the curvatures are calculated

    :rtype: 1d array
    :return: curvature evaluated at each point
    """
    x = points[:, 0]
    y = points[:, 1]

    # Velocity
    dx = np.gradient(x)
    dy = np.gradient(y)
    # Speed
    ds = np.sqrt(dx**2 + dy**2)

    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curv = np.abs(d2x * dy - dx * d2y) / ds**3
    if returnMean:
        return curv.mean()
    else:
        return curv

points = np.vstack([resid, resid5]).T
curvature(points)

points = np.vstack([resid, resid]).T
curvature(points)