# This script is used to test if the sequential OKGT works.

# Set the working directory to "/Users/panc25/Dropbox/Research/Zhu_Michael/my_paper/Paper_1/code"
# Rename the KOT class file to "okgt_class.py"

from scipy.linalg import sqrtm
eps = 0.001

# Load my KOT class script
%run ./core/okgt_class.py 

# Simulate data
# Model:
#	Y = tan(sin(x1) + x2^2)
np.random.seed(10)
n = 500 
p = 2
# X = np.matrix(np.random.randn(n,p)) # X is a numpy matrix
X = np.matrix(np.random.uniform(-1,1, (n,p)))
# Y = np.tan( np.sin(X[:,0]) + np.power(X[:,1],2) ) # Y is also a numpy matrix
Y = abs( np.sin(np.pi * X[:,0]) + np.power(X[:,1],2) ) 

# Set parameters
x1KernelName = ['Laplace']; x1KernelPara = [dict(alpha = 0.5)]
x2KernelName = ['Laplace']; x2KernelPara = [dict(alpha = 0.5)]
yKernelName = ['Laplace']; yKernelPara = [dict(alpha = 0.5)]

# Apply KOT on Y~X1
kot1 = KOT(X[:,0], Y, x1KernelName, yKernelName, x1KernelPara, yKerenlPara, eps=1e-3)
g1_hat = kot1.g
f1_hat = kot1.f 

# Construct gram matrices
kern = kot1.xKernel_fns[0]
Kx1 = KOT.GramMtx(X[:,0], kern)
Kx2 = KOT.GramMtx(X[:,1], kern)
Ky = KOT.GramMtx(Y, kern)

# Kx2_oc spans the orthogonal complement of {f2 - f_{2|1}}, where f_{2|1} is the projection of f_2 onto H_x2.
Kx1Kx1_inv = np.linalg.inv(Kx1 * Kx1)
A = Kx2 * Kx1 * Kx1Kx1_inv * Kx1 * Kx2
I = np.identity(n)
Kx2_oc = I - A # this is the new Kx2, which is used to apply KOT against Y.

## USe Kx2_oc as the new basis, run KOT against Y
Kx2_oc = sqrtm( Kx2_oc.T * Kx2_oc ) # make Kx2_oc symmetric
Kx = Kx2_oc
Ky_inv = np.linalg.inv( sqrtm( Ky.T*Ky ) ) # Ryy^{-1/2}

Ryy = Ky.T*Ky
D, P = np.linalg.eigh(Ryy + eps * np.identity(n))
D = D[::-1]
P = P[:, ::-1]
D_inv = np.matrix(np.diag(1./np.sqrt(D)))
Ky_inv = D_inv * P.T # Ryy^{-1/2}

Ryx = Ky * Kx
Rxx = Kx * Kx
Vyx2Vx2y = Ky_inv * Ryx * np.linalg.inv(Rxx + eps * np.identity(n)) * Ryx.T * Ky_inv.T

r2, beta = slinalg.eigh(Vyx2Vx2y, eigvals=(n-1, n-1))
beta = np.matrix(beta)
zeta = D_inv * beta
zeta = P * zeta 
g2_hat = Ky * zeta

xi = Ryx.T * g2_hat
xi = np.linalg.inv(Rxx + eps * np.identity(n)) * xi
f2_hat = Kx * xi # marginal f2 in the orthogonal space contained in H_x2

g_hat = g1_hat + g2_hat

## Adjust f2_hat 
f1_hat_adj = np.linalg.inv(Kx1 * Kx1 + eps * I) * Kx1 * Kx2 * f2_hat

# Plot
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(2,3)
fig.tight_layout()
axarr[0,0].scatter(np.array(Y), np.array(g_hat), s=0.5)
axarr[0,0].set_title('g')
axarr[0,1].scatter(np.array(Y), np.array(g1_hat), s=0.5)
axarr[0,1].set_title('g1')
axarr[0,2].scatter(np.array(Y), np.array(g2_hat), s=0.5)
axarr[0,2].set_title('g2')
axarr[1,0].scatter(np.array(X[:,0]), np.array(f1_hat), s=0.5)
axarr[1,0].set_title('f1, marginal')
axarr[1,1].scatter(np.array(X[:,1]), np.array(f2_hat), s=0.5)
axarr[1,1].set_title('f2, orthogonal')
plt.show()

# TODO: 
# 1) Plot the x2 ~ f_{2|1}
# 2) Chnage distributions to Unif(-3, 3)

# ================================================
# 3-11-2015 

# Implement the sequential algorithm when there are two predictor variables

import numpy as np
from scipy.linalg import sqrtm
from sklearn import linear_model
import statsmodels.api as sm # for weighted least square

## Some constants
eps = 0.001 # regularization parameter
tol = 0.001 # tolerance for iteration

## Simulate data
## Model: Y = exp( sin(2pi * X1) + X2^2 ),
## with no noise.
np.random.seed(10)
n = 500 # sample size
p = 2 # number of predictors
X = np.matrix(np.random.uniform(-1, 1, (n,p)))
Y = np.exp(np.sin(np.pi * X[:,0]) + np.power(X[:,1],2)) 

## Construct centered Gram matrices
%run /Users/panc25/Dropbox/Research/Zhu_Michael/my_paper/Paper_1/code/core/okgt_class.py

kernName = ['Laplace']
kernPara = [dict(alpha = 0.5)]
# kernName = ['Gaussian']
# kernPara = [dict(sigma = 0.5)]
kernFun = KOT.ConstructKernelFns(kernName, kernPara)[0]

Ky = KOT.GramMtx(Y, kernFun)
Kx1 = KOT.GramMtx(X[:,0], kernFun)
Kx2 = KOT.GramMtx(X[:,1], kernFun)

## Initialize g^(1) using OKGT 
# kot0 = KOT(X[:,0], Y, 
# 	xKernelNames=kernName, yKernelName=kernName, 
# 	xKernelParas=kernPara, yKernelPara=kernPara, 
# 	eps=eps)

# g = kot0.g.real # TODO: modify the code to produce real numbers as the output.
# g = g/np.std(g) # TODO: why my g is not of unit length?

## Initialize using LS: X1 ~ Ky.
# regr = linear_model.LinearRegression()
# regr.fit(Ky, X[:,0])
# alpha = regr.coef_.T
# g = Ky * alpha
# g = g / np.std(g)

## Initialize using standardized Y
g = (Y - np.mean(Y))/np.std(Y)

## The following matrices doesn't change in iteration.
## Since K's are all symmetric, no need to consider transpose.
P0 = np.linalg.inv(Kx1 * Kx1 + eps * np.identity(n)) * Kx1
P = Kx1 * P0
R = np.identity(n) - P
L = P0 * Kx2

## Iteration
count = 0
while count < 30:
	count += 1 
	### Create linear regression object
	regr = linear_model.LinearRegression()
	regr.fit(Kx1, g)
	beta0 = regr.coef_.T

	beta2 = R * g
	beta2 = Kx2 * beta2
	beta2 = np.linalg.inv(Kx2 * R * Kx2 + eps * np.identity(n)) * beta2
	### TODO: the above three line is actually a GLS problem, find if a package exists which would probably more efficient.

	beta1 = beta0 - L * beta2

	f1 = Kx1 * beta1
	f2 = Kx2 * beta2

	### MSE
	e2 = np.var(g - f1 - f2)
	print "Iteration", count, "\t: e2 = ", e2

	regr.fit(Ky, f1+f2)
	alpha = regr.coef_.T
	g = Ky * alpha
	g = g / np.std(g)

	### Update g as f1+f2
	# g = f1 + f2
	# g = g / np.std(g)

# -------------------------------------------------
# Plot
## Plot the transformation after 3 iterations
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(1,3)
fig.tight_layout()
axarr[0].scatter(Y, g, s=0.5)
axarr[1].scatter(X[:,0], f1, s=0.5)
axarr[2].scatter(X[:,1], f2, s=0.5)
plt.show()
