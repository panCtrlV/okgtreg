__author__ = 'panc'

import numpy as np
from okgtreg.Kernel import Kernel

# Test Kernel.kernelMapping
kernel = Kernel('gaussian', sigma=0.5)
## mapping at a scalar value
phi = kernel.kernelMapping(0.5)
phi(0.5)
## mapping at a vector
phi = kernel.kernelMapping(np.array([0.5, 0.5]))
phi(np.array([0.5, 0.5]))
phi(np.array([0.5, 1.]))

# Test Kernel.kernelSpan
coef = np.array([0.1, 3., -0.5])
x = np.array([[0, .5, .3],
              [-0.3, 0.4, 0.2],
              [0.4, 0.9, 1.]])
f = kernel.kernelSpan(x, coef)
## Evaluate the span at a single point
y1 = np.array([0, .5, .3])
f(y1)
## Evaluate the span at multiple points
y2 = np.array([[0, .5, .3],
               [-0.3, 0.4, 0.2]])
f(y2)
## x and y are both one-dimensional
x = np.array([[-0.5], [0.], [0.5]])
y1 = np.array([1.])
f2 = kernel.kernelSpan(x, coef)
f2(y1)

y2 = np.array([[1.], [2.]])
f2(y2)
## Wrong input format for x
x = np.array([[-0.5, 0., 0.5]])
f = kernel.kernelSpan(x, coef)
f(y1)

# Test ParameterizedData.getXFromGroup
from okgtreg.Data import Data, ParameterizedData
from okgtreg.Kernel import Kernel
from okgtreg.Group import Group
from okgtreg.Parameters import Parameters


## simulate data
### model 3
### fully additive
### h = 2*x1 + x2**2 + x3**3 + sin(x4*pi) + log(x5+5) + |x6|
### y = ln(h^2)
def model3(n):
    p = 6
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,)) * 0.01
    h = 2. * x[:, 0] + \
        x[:, 1] ** 2 + \
        x[:, 2] ** 3 + \
        np.sin(x[:, 3] * np.pi) + \
        np.log(np.abs(x[:, 4] + 5.)) + \
        np.abs(x[:, 5]) + \
        e
    y = np.log(h ** 2)
    return Data(y, x), Group([1], [2], [3], [4], [5], [6]), h


data, group, h = model3(500)

kernel = Kernel('gaussian', sigma=0.5)

parameters = Parameters(group, kernel, [kernel] * group.size)
parameterizedData = ParameterizedData(data, parameters)

parameterizedData.getXFromGroup(4)  # return 2d array, even for a single data column

#################################
# Test OKGTReg2._train_Vanilla2 #
#################################
# Now the f transformations are also returned as callables
import time
import numpy as np
import matplotlib.pyplot as plt
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.Kernel import Kernel
from okgtreg.Data import Data, ParameterizedData
from okgtreg.Group import Group
from okgtreg.Parameters import Parameters


## Data Simulation
### model 3
### fully additive
### h = 2*x1 + x2**2 + x3**3 + sin(x4*pi) + log(x5+5) + |x6|
### y = ln(h^2)
def model3(n):
    p = 6
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,)) * 0.01
    h = 2. * x[:, 0] + \
        x[:, 1] ** 2 + \
        x[:, 2] ** 3 + \
        np.sin(x[:, 3] * np.pi) + \
        np.log(np.abs(x[:, 4] + 5.)) + \
        np.abs(x[:, 5]) + \
        e
    y = np.log(h ** 2)
    return Data(y, x), Group([1], [2], [3], [4], [5], [6]), h


### generate data
np.random.seed(25)
data, group, h = model3(500)

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# OKGT
# okgt = OKGTReg2(data, kernel=kernel, group=group, eps=1e-6)
# okgt = OKGTReg2(data, kernel=kernel, group=Group([1,2,3,4,5,6]), eps=1e-6)
okgt = OKGTReg2(data, kernel=kernel, group=Group([1, 2, 3], [4, 5, 6]), eps=1e-6)
# okgt = OKGTReg2(data, kernel=kernel, group=Group([1,2], [3,4], [5,6]), eps=1e-6)
## Fit
start = time.time()
# fit = okgt._train_Vanilla2(h)
fit = okgt._train_lr(h)
stop = time.time()
print 'R2 =', fit['r2']
print 'Elapsed time =', stop - start, 'sec'
## Plot
### g
# plt.scatter(data.y, fit['g'])
### fitted f values (the old output)
# fig, axarr = plt.subplots(2,3)
# for i in range(2):
#     for j in range(3):
#         idx = i*3 + j
#         axarr[i,j].scatter(data.X[:,idx], fit['f'][:,idx], s=0.8)
### predicted values for h
# h_pred_list = [fit['f_call'][i](okgt.parameterizedData.getXFromGroup(i)) for i in range(1,7)]
# h_pred = np.column_stack(h_pred_list).sum(axis=1)
# plt.scatter(h, h_pred)

### Now we generate a test set, and evaluate the prediction error
np.random.seed(3)
test_data, test_group, test_h = model3(500)
# test_group = Group([1,2,3,4,5,6]) # not true group structure
test_group = Group([1, 2, 3], [4, 5, 6])
# test_group = Group([1,2], [3,4], [5,6])
parameters = Parameters(test_group, kernel, [kernel] * test_group.size)
parameterizedTestData = ParameterizedData(test_data, parameters)
test_h_pred_list = [fit['f_call'][i](parameterizedTestData.getXFromGroup(i)) for i in range(1, test_group.size + 1)]
test_h_pred = np.column_stack(test_h_pred_list).sum(axis=1)
# plt.scatter(test_h, test_h_pred)
test_R2 = 1 - sum((test_h - test_h_pred) ** 2) / sum((test_h - np.mean(test_h)) ** 2)
print "Test R2 =", test_R2

## using the f callables (new output)
# x = np.linspace(-3., 3., 100)
# fig, axarr = plt.subplots(2,3)
# for i in range(2):
#     for j in range(3):
#         f_fit = fit['f_call'][i*3+j+1](x[:,np.newaxis])
#         # f_fit_norm = np.linalg.norm(f_fit, ord=2)
#         axarr[i,j].scatter(x, f_fit, s=0.8)
