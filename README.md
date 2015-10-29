Optimal Kernel Group Transformation for Exploratory Regression and Graphics Python Package
==========================================================================================

# Introduction

This Python package implements our paper accepted by SIGKDD 2015 (ID: fp410)

# Installation

`pip install okgtreg` 

# Usage

```python
from okgtreg.core import *

"""
p = 5
n = 500
l = 5
"""

y, X = DataSimulator.SimData_Wang04(500)  # Simulate data
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

# Reference

[Chao, Pan, Qiming Huang, and Michael Zhu. "Optimal Kernel Group Transformation for Exploratory Regression Analysis and Graphics." In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 905-914. ACM, 2015.](http://www.stat.purdue.edu/~panc/research/publication/okgt_paper.pdf)  