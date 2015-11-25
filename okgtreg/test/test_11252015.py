from okgtreg.DataSimulator import DataSimulator
from okgtreg.Group import Group
from okgtreg.Kernel import Kernel
from okgtreg.Parameters import Parameters


data = DataSimulator.SimData_Wang04WithInteraction(100)
group = Group([1], [2], [3], [4], [5], [6, 7])
kernel = Kernel('gaussian', sigma=0.5)
parameters = Parameters(group, kernel, [kernel]*group.size)
