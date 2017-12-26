Optimal Kernel Group Transformation for Exploratory Regression and Graphics Python Package
==========================================================================================

# Introduction

This Python package implements our KDD 2015 and NIPS 2017 papers.

The main functionality of this package is to 
- estimate an OKGT given a data set, desired group structure and kernel functions. 
- estimate the optional group structure if the desired group structure is not known.

~~# Installation~~

~~`pip install okgtreg`~~

# Usage

```python
from okgtreg import *

"""
p = 5
n = 500
l = 5
"""

data_simulator = DataSimulator(seed=123)
y, X = data_simulator.SimData_Wang04(500)  # Simulate data
data = Data(y, X)  # construct data object
group = Group([1], [2], [3], [4], [5])  # construct group object
ykernel = Kernel('gaussian', sigma=0.1)
xkernels = [Kernel('gaussian', sigma=0.5)]*5
parameters = Parameters(group, ykernel, xkernels)  # construct parameters object
# parameterizedData = ParameterizedData(data, parameters)

okgt = OKGTReg(data, parameters)  # construct okgt object
res = okgt.train_Vanilla()  # training

import matplotlib.pyplot as plt
plt.scatter(y, res['g'])
j=4
plt.scatter(X[:, j], res['f'][:, j])
```

## Example of using forward and backward selection procedure to discover group structure

```python
from okgtreg.DataSimulator import *
from okgtreg.forwardSelection import *
from okgtreg.backwardSelection import *

# Simulate data
data_simulator = DataSimulator(seed=123)
y, x = data_simulator.SimData_Wang04WithInteraction(500)
data = Data(y, x)

# Same kernel for all groups
kernel = Kernel('gaussian', sigma=0.5)

# Forward selection (with low rank approximation for Gram matrix)
fGroup = forwardSelection(data, kernel, True, 10)

# Backward selection (with low rank approximation for Gram matrix)
bGroup = backwardSelection(data, kernel, True, 10)
```


# Reference

Pan Chao, Qiming Huang, and Michael Zhu. [Optimal Kernel Group Transformation for Exploratory Regression Analysis and Graphics.](http://www.stat.purdue.edu/~panc/research/publication/okgt_paper.pdf)  KDD 2015

Pan Chao and Michael Zhu. [Group Additive Structure Identification for Kernel Nonparametric Regression](http://papers.nips.cc/paper/7076-group-additive-structure-identification-for-kernel-nonparametric-regression.pdf%7D) NIPS 2017

