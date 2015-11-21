from okgtreg.DataSimulator import *
from okgtreg.Data import *
from okgtreg.Kernel import *
from okgtreg.Parameters import *

import scipy.linalg as slin

"""
Instead of using R2 as the selection criterion, this implementation
uses HS norm of the cross-covariance operator and joint cross-covariance
operator.

Motivated by R2 formula:

    R2 = Var(\hat{Y}) / Var(Y),

we can define the non-parametric version of R2 for OKGT as

    R2 = Cov( g(Y), \sum_j f_j(X_j) ) / Var(g(Y)).  --- (1)

The calculation of the covariance and variance can be done by using
cross-covariance operators as;

    Cov(g(Y), \sum_j {f_j(X_j)})
        = \sum_j Cov( g(Y), f_j(X_j) )
        = \sum_j < g, R_{YX_j}f_j >_{H_Y},      --- (2)

    Var(g(Y)) = < g, R_{YY}g >_{H_Y}.       --- (3)

If we use block operator notation, (2) could be simplified to:

    Cov(g(Y), \sum_j {f_j(X_j)}) = < g(Y), R^+_{YX} (f_1, f_2, ..., f_p) >_{H_Y}        --- (4)

where R^+_{YX} is the column stack of R_{YX_j}'s.

...
"""


# Simulate data
np.random.seed(25)

nSample = 500
y, x = DataSimulator.SimData_Wang04(nSample)
data = Data(y, x)
group = Group(*tuple([i] for i in np.arange(5) + 1))

kernel = Kernel("gaussian", sigma=0.5)

parameters = Parameters(group, kernel, [kernel]*group.size)
parameterizedData = ParameterizedData(data, parameters)


# Calculate covariance operator R_{YY}
Ryy, Ky = parameterizedData.covarianceOperatorForY(returnAll=True)
hsNorm2_y = np.diag(Ky.dot(Ky)).sum()
hsNorm2_y

sy, vy = slin.eigh(Ryy, eigvals=(nSample-1, nSample-1))
sy
vy


# Calculate R^+_{XX}
Rxx, Kx = parameterizedData.covarianceOperatorForX(returnAll=True)
hsNorm2_x = np.diag(Rxx).sum()
# hsNorm2_x = np.diag(Kx.dot(Kx.T)).sum() / 500
hsNorm2_x

sx, vx = slin.eigh(Rxx, eigvals=(nSample * group.size - 1, nSample * group.size - 1))
sx
vx


# Calculate cross-covariance operator R^+_{YX}
Ryx = parameterizedData.crossCovarianceOperator()
Ryx.shape

np.diag(Ryx.dot(Ryx.T)).sum()
s, v = slin.eigh(Ryx.dot(Ryx.T), eigvals=(nSample-1, nSample-1))
s



s / sx / sy
s, sx, sy


np.diag(Ryx.dot(Ryx.T)).sum(), np.diag(Rxx).sum(), np.diag(Ryy).sum()