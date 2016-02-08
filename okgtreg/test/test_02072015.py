__author__ = 'panc'

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Kernel import Kernel
from okgtreg.OKGTReg import OKGTReg2

# simulate data
data, group, h = DataSimulator.SimData_Wang04WithInteraction(500)

kernel = Kernel('gaussian', sigma=0.5)
okgt = OKGTReg2(data, kernel=kernel, group=group)
fit = okgt._train_Vanilla2(h)
print fit['r2']

import matplotlib.pyplot as plt

j = 4;
plt.scatter(data.X[:, j], fit['f'][:, j])
plt.scatter(h, fit['g'])

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 5], data.X[:, 6], fit['f'][:, 5])
