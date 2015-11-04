from core.Group import *
from core.DataSimulator import *
from core.okgtreg import *

#
# group = Group([1,2], [3], [4,5,6])
# group.partition
# newGroup = group.addNewCovariateAsGroup(7)
# newGroup.partition
# newGroup.p
#
# group = Group([1,2])
# group.partition
# newGroup = group.addNewCovariateToGroup(3, 1)
# newGroup.partition
# newGroup = group.addNewCovariateToGroup(4, 2)  # group number out of bound
# newGroup = group.addNewCovariateToGroup(1, 1)  # duplicate covariate
#
# group = Group()  # empty group
# group.addNewCovariateToGroup(1, 1)
# group.addNewCovariateAsGroup(1)
#
#
# group = Group([2,1], [7,4], [3,5,6])
#
# d, g = data.getGroupedData(Group([2,1], [4]))
# d.X
# g.partition


"""
While determining structure, kernel functions should be fixed.
"""
# Test forward selection
np.random.seed(25)
y, X = DataSimulator.SimData_Wang04(1000)  # Simulate data
data = Data(y, X)
ykernel = Kernel('gaussian', sigma=0.5)
kernel = Kernel('gaussian', sigma=0.5)

covariatesPool = list(np.arange(data.p) + 1)
oldGroup = Group()
bestR2 = 0.
bestOKGT = None
bestCovariateIndex = None
bestGroupIndex = None

while len(covariatesPool):
    print "** Available covariates: ", covariatesPool
    # add a new group no matter what
    print "** Add as new group: **"
    for covariateInd in covariatesPool:
        print("\t try covariate %d ..." % covariateInd)
        currentGroup = oldGroup.addNewCovariateAsGroup(covariateInd)
        print("\t\t current group structure: %s " % (currentGroup.partition,))
        # The following OKGT needs a subset of data and the grouped covariate
        # indices being normalized, so that the training is done as if we are
        # using a complete data.
        dataForOKGT, groupForOKGT = data.getGroupedData(currentGroup)
        print groupForOKGT.partition
        # TODO: Currently, using same kernel for all groups.
        # todo: Is it possible to adapt kernels to different group structure?
        xkernels = [kernel] * groupForOKGT.size
        parametersForOKGT = Parameters(groupForOKGT, ykernel, xkernels)
        currentOKGT = OKGTReg(dataForOKGT, parametersForOKGT)
        # res = currentOKGT.train_Vanilla()
        res = currentOKGT.train_Nystroem(10)
        currentR2 = res['r2']
        if currentR2 > bestR2:
            print("\t\t current R2 =\t %.10f \t *" % currentR2)
            bestR2 = currentR2
            bestOKGT = currentOKGT
            bestCovariateIndex = covariateInd
            newGroup = currentGroup
        else:
            print("\t\t current R2 =\t %.10f" % currentR2)
        print("\t\t best R2 =\t\t %.10f" % bestR2)
    print("** updated group structure is: %s" % (newGroup.partition, ))
    # if group structure is not empty, a new covariate can be added to an existing group
    # print oldGroup.size
    if oldGroup.size is not 0:
        print "** Add to an existing group: **"
        # can add new covariate to existing group
        for covariateInd in covariatesPool:
            for groupInd in np.arange(oldGroup.size)+1:
                # print oldGroup.partition
                print("\t try adding covariate %d " % covariateInd)
                print("\t in group %d ..." % groupInd)
                currentGroup = oldGroup.addNewCovariateToGroup(covariateInd, groupInd)
                print("\t\t current group structure: %s " % (currentGroup.partition,))
                # print currentGroup.partition
                dataForOKGT, groupForOKGT = data.getGroupedData(currentGroup)
                print groupForOKGT.partition
                xkernels = [kernel] * groupForOKGT.size
                parametersForOKGT = Parameters(groupForOKGT, ykernel, xkernels)
                currentOKGT = OKGTReg(dataForOKGT, parametersForOKGT)
                # res = currentOKGT.train_Vanilla()
                res = currentOKGT.train_Nystroem(10)
                currentR2 = res['r2']
                if currentR2 > bestR2:
                    print("\t\t current R2 =\t %.10f \t *" % currentR2)
                    bestR2 = currentR2
                    bestOKGT = currentOKGT
                    bestCovariateIndex = covariateInd
                    bestGroupIndex = groupInd
                    newGroup = currentGroup
                else:
                    print("\t\t current R2 =\t %.10f" % currentR2)
                print("\t\t best R2 =\t\t %.10f" % bestR2)
    print("** updated group structure is: %s \n" % (newGroup.partition, ))
    covariatesPool.remove(bestCovariateIndex)  # TODO: update in-place, good?
    oldGroup = newGroup

"""
Using true structure
"""
group = Group([1], [2], [3], [4], [5])
xkernels = [kernel] * 5
parameters = Parameters(group, ykernel, xkernels)
okgt = OKGTReg(data, parameters)
res = okgt.train_Vanilla()

res['r2']

res['g']

import matplotlib.pyplot as plt

plt.scatter(y, res['g'])
j=4; plt.scatter(X[:, j], res['f'][:, j])


resid = res['g'] - res['f'].sum(axis=1)[:, np.newaxis]
plt.plot(resid)
np.var(resid) / np.var(res['g'])


# """
# Old implementation
# """
# import okgtreg_primitive.okgtreg_primitive as oldokgt
#
# n, p = X.shape
# kname = 'Gaussian'
# kparam = dict(sigma=0.5)
# okgt_old = oldokgt.OKGTReg(X, y[:, np.newaxis], [kname]*p, [kname], [kparam]*p, [kparam])
# res_old = okgt_old.TrainOKGT()
#
# res_old[2]  # R2
#
# res_old[0]  # g
#
# import matplotlib.pyplot as plt
#
# plt.scatter(y, res['g'])
# plt.scatter(y, np.array(res_old[0]))