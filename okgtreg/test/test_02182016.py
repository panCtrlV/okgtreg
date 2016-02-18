__author__ = 'panc'

from okgtreg import *
import matplotlib.pyplot as plt

import numpy as np
import time

np.random.seed(25)
data, group, h = DataSimulator.SimData_Wang04WithInteraction(500)
kernel = Kernel('gaussian', sigma=0.5)
# okgt = OKGTReg2(data, kernel=kernel, group=group)
# okgt = OKGTReg2(data, kernel=kernel, group=Group([1,2,3,4,5,6,7]))
okgt = OKGTReg2(data, kernel=kernel, group=Group([1], [2], [3], [4], [5], [6], [7]))

start = time.time()
fit_lr = okgt._train_lr(h)
stop = time.time()
print "time: ", stop - start
print fit_lr['r2']

# j=5; plt.scatter(data.X[:,j], fit_lr['f'][:,j])

start = time.time()
fit_vanilla = okgt._train_Vanilla2(h)
stop = time.time()
print "time: ", stop - start
print fit_vanilla['r2']
