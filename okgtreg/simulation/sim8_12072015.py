import numpy as np

from okgtreg.Data import Data
from okgtreg.Group import RandomGroup, Group
from okgtreg.Kernel import Kernel
from okgtreg.groupStructureDetection.splitAndMergeWithRandomInitial import splitAndMergeWithRandomInitial, splitAndMergeWithRandomInitial2
from okgtreg.OKGTReg import OKGTReg
from okgtreg.Parameters import Parameters

# Simple data generator
def simulateData(n):
    x = np.random.uniform(-1., 1., (n, 4))
    noise = np.random.normal(0., 1., n) * 0.1
    y = x[:,0]**2 + np.sin(x[:, 1]) + x[:, 2] * x[:, 3] + noise
    return Data(y, x)

# Generate data
np.random.seed(25)
data = simulateData(1000)

# Same kernel
kernel = Kernel('gaussian', sigma=0.5)

# Random group as initial
# group0 = RandomGroup(2, (1,2,3,4))

# Group structure detection (with conservative split)
optimalOkgt = splitAndMergeWithRandomInitial2(data, kernel, True, 10, 25)
optimalOkgt = splitAndMergeWithRandomInitial2(data, kernel, True, 10, 125)
optimalOkgt = splitAndMergeWithRandomInitial2(data, kernel, True, 10, 1125)

# Group Structure detection (with aggressive split)
optimalOkgt = splitAndMergeWithRandomInitial(data, kernel, True, 10, 25)

optimalOkgt.r2

group = Group([1], [2], [3,4])
parameters = Parameters(group, kernel, [kernel]*group.size)
okgt = OKGTReg(data, parameters)
okgt.train('nystroem', 10)

# True okgt
trueGroup = Group([1], [2], [3, 4])
parameters = Parameters(trueGroup, kernel, [kernel]*trueGroup.size)
trueOkgt = OKGTReg(data, parameters)
fit = trueOkgt.train('nystroem', 10, 25)
fit['r2']