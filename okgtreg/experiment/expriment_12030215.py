"""
In this experiment, I want to text if the optimal transformations of the covariates
have some correlation structure after fitting an OKGT with fully additive structure.

The experiment is conducted as follows:

1. Simulate data from a model which has a group structure. That is, some covariate
   belong to the same group, while others each forms an univariate group.

2. Fit OKGT on the simulated data with a fully additive model.

3. Calculate correlation matrix for the transformed covariates.
"""

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Kernel import Kernel
from okgtreg.Group import Group
from okgtreg.Parameters import Parameters
from okgtreg.OKGTReg import OKGTReg
from okgtreg.Data import Data

import pandas as pd
from pandas import DataFrame
import numpy as np


# Simulate data
n = 500

np.random.seed(25)
data = DataSimulator.SimData_Wang04WithInteraction2(n)
# data = DataSimulator.SimData_Wang04WithInteraction(n)

# Standardize data
dfX = DataFrame(data.X)
dfX_stand = (dfX - dfX.mean()) / dfX.std()
X_stand = np.array(dfX_stand)

dfy = DataFrame(data.y)
dfy_stand = (dfy - dfy.mean()) / dfy.std()
y_stand = np.array(dfy_stand)
y_stand.shape = (n, )

[dfy_stand[0].corr(dfX[i]) for i in range(data.p)]

data_stand = Data(y_stand, X_stand)

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Group structure used to fit OKGT
group = Group(*tuple([[i+1] for i in range(data.p)]))
# group = Group([1], [2], [3], [4], [5], [6, 7, 8])

# Construct parameters with kernel and group
parameters = Parameters(group, kernel, [kernel]*group.size)

# Fit OKGT
# okgt = OKGTReg(data, parameters)
okgt_stand = OKGTReg(data_stand, parameters)
# fit = okgt.train('nystroem', 10, 25)
# fit = okgt.train()
fit_stand = okgt_stand.train()
# fit['r2']
fit_stand['r2']

# Correlation matrix for the transformed covariates
# df = DataFrame(fit['f'])
df = DataFrame(fit_stand['f'])
df.corr()
df.cov()

df2 = DataFrame(data.X)
df2.corr()

# Correlation between f_\ell(x_\ell) and g(y)
# [DataFrame(fit['g'])[0].corr(df[i]) for i in range(group.size)]
[DataFrame(fit_stand['g'])[0].corr(df[i]) for i in range(group.size)]

# Plot
import matplotlib.pyplot as plt

plt.scatter(data.X[:,5], fit['f'][:,5])

# # Conditional covariance
# f = fit['f']
# g = fit['g']
#
# Rxx = f.T.dot(f) / n
# DataFrame(Rxx)
# Rxy = f.T.dot(g) / n
# Rxy.shape = (group.size, 1)
# Ryy = g.T.dot(g) / n
# Rxxy = Rxx - Rxy.dot(Rxy.T) / Ryy
# DataFrame(Rxxy)


""" Observations:
The conditional variances of X given Y ara small for those covariates sharing the groups.
Is this true in general? Tried the group with three following two group structures.
It is true at least for these two cases.

    1) Group structure with a tri-variate group: ([1], [2], [3], [4], [5], [6, 7, 8])

                0          1          2          3           4         5  \
    0  218.435180   8.054959  -5.185223   2.408891    1.764084  0.180784
    1    8.054959  49.621383   0.877104  -0.542316    5.436549  0.091346
    2   -5.185223   0.877104  45.823505   0.283053   -3.606470 -0.261644
    3    2.408891  -0.542316   0.283053  74.109595   -2.126183  0.073937
    4    1.764084   5.436549  -3.606470  -2.126183  173.523797 -0.039677
    5    0.180784   0.091346  -0.261644   0.073937   -0.039677  0.417648
    6   -0.719203  -0.103129   0.066479  -0.273512   -0.177834  0.010765
    7    0.558253  -0.100011   0.198169   0.494165   -0.087735  0.018742

              6         7
    0 -0.719203  0.558253
    1 -0.103129 -0.100011
    2  0.066479  0.198169
    3 -0.273512  0.494165
    4 -0.177834 -0.087735
    5  0.010765  0.018742
    6  0.416884  0.006397
    7  0.006397  0.686303


    2) Group structure with a bi-variate group: ([1], [2], [3], [4], [5], [6, 7])

                0          1          2          3           4         5             6
    0  276.418358   2.581728  -3.294919   5.082582    8.210694  -2.529011       -0.568543
    1    2.581728  57.085757  -3.942816   5.112468   -2.336414  -0.595938        0.292360
    2   -3.294919  -3.942816  53.744036  -0.595958   -3.792824   0.512295       -0.505247
    3    5.082582   5.112468  -0.595958  78.716448   -0.778319   0.241946       -0.391473
    4    8.210694  -2.336414  -3.792824  -0.778319  213.843100  -1.036675        0.870721
    5   -2.529011  -0.595938   0.512295   0.241946   -1.036675  **2.607612**    -0.051640
    6   -0.568543   0.292360  -0.505247  -0.391473    0.870721  -0.051640       **0.824114**


When I fit the OKGT on the data simulated from the first groups structure with the true structure,
the covariance for the tri-variate transformation is also small.

                0          1          2          3           4         5
    0  214.287272   7.701545  -5.003764   2.382480    1.724198  2.245828
    1    7.701545  45.866391   0.709382  -0.377360    5.261856  1.729058
    2   -5.003764   0.709382  44.582546   0.393316   -3.454518  0.648990
    3    2.382480  -0.377360   0.393316  71.834767   -2.249219  1.376296
    4    1.724198   5.261856  -3.454518  -2.249219  169.098740  2.046092
    5    2.245828   1.729058   0.648990   1.376296    2.046092  6.578028


Then, I changed the model to:

    y = np.log( 4.0 +
                np.sin(4 * x[:, 0]) +
                np.abs(x[:, 1]) +
                x[:, 2]**2 +
                x[:, 3]**3 +
                x[:, 4] +
                100. * abs(x[:, 5] * x[:, 6] * x[:, 7]) +
                0.1 * noise).

That is, the tri-variate transformation is multiplied by 100 to increase its magnitude.
By fiiting the simulated data with a fully additive group structure, we have the following
covariance matrix:

                 0            1           2            3            4  \
    0  1409.923895   121.808352  -38.963742   -32.168875   -90.115435
    1   121.808352  1732.090516   16.437021   -36.224481    94.424016
    2   -38.963742    16.437021  648.011489   -78.050360  -142.888206
    3   -32.168875   -36.224481  -78.050360  1119.396253   -30.755318
    4   -90.115435    94.424016 -142.888206   -30.755318  3111.298189
    5  -950.882604  -965.155636  -95.127980   516.383514   336.527013
    6  -295.094510  1345.936921  243.827882    62.324447  -339.470295
    7  1075.264473   171.099115 -442.607409   750.334420  -191.386300

                  5             6             7
    0   -950.882604   -295.094510   1075.264473
    1   -965.155636   1345.936921    171.099115
    2    -95.127980    243.827882   -442.607409
    3    516.383514     62.324447    750.334420
    4    336.527013   -339.470295   -191.386300
    5  90469.316149  -4288.636999   8162.159206
    6  -4288.636999  84970.121323  -2319.001865
    7   8162.159206  -2319.001865  87327.693626

which is against the observation we had before. This time the variance of the last three
transformed covariates are actually much larger than the other covariates.
"""


# TODO: Found magnitude of the original data will affect the correlation between g and each f.
# todo: For example, by multiplying the tri-variate group by 100, the correlations between the
# todo: transformation of each of those three covariates and g are increased. This is true even
# todo: after the original data are standardized.