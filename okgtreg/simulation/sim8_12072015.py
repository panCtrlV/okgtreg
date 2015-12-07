import numpy as np

from okgtreg.Data import Data
from okgtreg.Group import RandomGroup, Group
from okgtreg.Kernel import Kernel
from okgtreg.groupStructureDetection.splitAndMergeWithRandomInitial import splitAndMergeWithRandomInitial
from okgtreg.OKGTReg import OKGTReg
from okgtreg.Parameters import Parameters

# Simpmle data generator
def simulateData(n):
    x = np.random.uniform(-1., 1., (n, 4))
    noise = np.random.normal(0., 1., n) * 0.1
    y = x[:,0]**2 + np.sin(x[:, 1]) + x[:, 2] * x[:, 3] + noise
    return Data(y, x)

# Generate data
data = simulateData(500)

# Same kernel
kernel = Kernel('gaussian', sigma=0.5)

# Random group as initial
group0 = RandomGroup(2, (1,2,3,4))

# Group structure detection
optimalOkgt = splitAndMergeWithRandomInitial(25, data, kernel, True, 10)
optimalOkgt.r2

# True okgt
trueGroup = Group([1], [2], [3, 4])
parameters = Parameters(trueGroup, kernel, [kernel]*trueGroup.size)
trueOkgt = OKGTReg(data, parameters)
fit = trueOkgt.train('nystroem', 10, 25)
fit['r2']