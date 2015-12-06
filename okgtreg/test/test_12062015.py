"""
Test okgtreg.OKGTRegForDetermineGroupStructure
"""

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Group import Group
from okgtreg.Kernel import Kernel
from okgtreg.Parameters import Parameters
from okgtreg.OKGTReg import OKGTRegForDetermineGroupStructure


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