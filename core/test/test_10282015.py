from core.okgtreg import *
from core.DataSimulator import *

# simulate
X = np.arange(50).reshape((10, 5))
y = np.arange(10)

# Data object
data = Data(y, X)

# Parameters object
group = Group([2,4,5], [1,3])
ykernel = Kernel('gaussian', sigma=0.1)
k1 = Kernel('gaussian', sigma=0.5)
k2 = Kernel('laplace', sigma=0.5)
xkernels = [k1, k2]
parameters = Parameters(group, ykernel, xkernels)

# ParameterizedData object
parameterizedData = ParameterizedData(data, parameters)

# Evaluate covariance and cross-covariance operators
cov = parameterizedData.covarianceOperator()
crossCov = parameterizedData.crossCovarianceOperator()

