import okgtreg_primitive.okgtreg as oldokgt
from core.okgtreg import *
from core.DataSimulator import *


# Simulate data
np.random.seed(25)
y, X = DataSimulator.SimData_Wang04(10)  # Simulate data
data = Data(y, X)
data.p


group = Group([1], [2], [3], [4], [5])

ykernel = Kernel('gaussian', sigma=0.5)
kernel = Kernel('gaussian', sigma=0.5)
xkernels = [kernel]*data.p

parameters = Parameters(group, ykernel, xkernels)

# OOP implementation
okgt = OKGTReg(data, parameters)
# res = okgt.train_Vanilla()
# res['r2']

# old implementation
kname = 'Gaussian'
kparam = dict(sigma=0.5)
okgt2 = oldokgt.OKGTReg(X, y[:, np.newaxis], [kname]*data.p, [kname], [kparam]*data.p, [kparam])
# res2 = okgt2.TrainOKGT()
# res2[2]