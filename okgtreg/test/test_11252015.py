from okgtreg.DataSimulator import DataSimulator
from okgtreg.Group import Group
from okgtreg.Kernel import Kernel
from okgtreg.Parameters import Parameters
from okgtreg.OKGTReg import OKGTReg


# Test OKGTReg.train
data = DataSimulator.SimData_Wang04WithInteraction(100)
group = Group([1], [2], [3], [4], [5], [6, 7])
kernel = Kernel('gaussian', sigma=0.5)
parameters = Parameters(group, kernel, [kernel]*group.size)
okgt = OKGTReg(data, parameters)

res1 = okgt.train(method='vanilla')
res1

res2 = okgt.train(method='nystroem')
res2


# Test Group._splitOneGroup
group = Group([1,2], [3,4,5], [6])
group._splitOneGroup(1)
group._splitOneGroup(2)
res = group._splitOneGroup(3)  # return None


# test OKGTReg.splitOptimalGroup
data = DataSimulator.SimData_Wang04WithInteraction(100)
group = Group([1,2], [3], [4], [5], [6,7])
kernel = Kernel('gaussian', sigma=0.5)
parameters = Parameters(group, kernel, [kernel]*group.size)
okgt = OKGTReg(data, parameters)

res = okgt.train(method='vanilla')
res['r2']

okgt2 = okgt.splitOptimalGroup(kernel)
res2 = okgt2.train(method='vanilla')
res2['r2']
