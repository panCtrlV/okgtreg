"""
Test okgtreg.OKGTRegForDetermineGroupStructure
"""

import numpy as np

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Group import Group
from okgtreg.Kernel import Kernel
from okgtreg.Parameters import Parameters
from okgtreg.OKGTReg import OKGTRegForDetermineGroupStructure


np.random.seed(25)
data = DataSimulator.SimData_Wang04(500)
group = Group(*tuple([i+1] for i in range(data.p)))
kernel = Kernel('gaussian', sigma=0.5)
parameters = Parameters(group, kernel, [kernel]*group.size)

# Check if the class can be executed successfully
okgt = OKGTRegForDetermineGroupStructure(data, parameters)
okgt.train()
okgt.r2
okgt.f.shape
okgt.g.shape

# Check .optimalSplit
group2 = Group([1], [2, 3, 4], [5])
parameters2 = Parameters(group2, kernel, [kernel]*group2.size)

okgt2 = OKGTRegForDetermineGroupStructure(data, parameters2)
okgt2.train('nystroem', 10, 25)
okgt2.r2

updateBySplit = okgt2.optimalSplit(kernel, 'nystroem', 10, 25)
updateBySplit.r2

# Check .optimalMerge
group3 = group
parameters3 = Parameters(group3, kernel, [kernel]*group3.size)

okgt3 = OKGTRegForDetermineGroupStructure(data, parameters3)
okgt3.train('nystroem', 10, 25)
okgt3.r2

updateByMerge = okgt3.optimalMerge(kernel, 'nystroem', 10, 25)
