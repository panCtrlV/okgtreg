"""
The results of experiment is given below:

1. Forward selection:

    ([1], [2], [3], [4], [5, 10, 15], [6, 8], [7, 17, 18], [9, 16], [11], [12], [13], [14])

    with R2 = 0.5586320233.

2. Backward determination:

    ([1], [2], [3], [4], [5, 10, 15], [6, 8], [7, 17, 18], [9, 16], [11], [12], [13], [14])

    with R2 = 0.558632.

3. Split-and-merger:

    ([1], [2], [3], [4], [5, 6, 10, 15], [7, 17, 18], [8], [9, 16], [11], [12], [13], [14])

    with R2 = R2 = 0.5586.

"""

from okgtreg.DataUtils import readSkillCraft1
from okgtreg.Kernel import Kernel
from okgtreg.Group import Group


data = readSkillCraft1()


# Forward group
groupTuple = ([1], [2], [3], [4], [5, 10, 15], [6, 8], [7, 17, 18], [9, 16], [11], [12], [13], [14])
groupForward = Group(*groupTuple)
groupedNames = data.getGroupedNames(groupForward)
for names in groupedNames:
    print names
"""
['Age']
['HoursPerWeek']
['TotalHours']
['APM']
['SelectByHotkeys', 'NumberOfPACs', 'WorkersMade']
['AssignToHotkeys', 'MinimapAttacks']
['UniqueHotkeys', 'ComplexUnitsMade', 'ComplexAbilitiesUsed']
['MinimapRightClicks', 'UniqueUnitsMade']
['GapBetweenPACs']
['ActionLatency']
['ActionsInPAC']
['TotalMapExplored']
"""

# Backward group, same as forward
groupTuple = ([1], [2], [3], [4], [5, 10, 15], [6, 8], [7, 17, 18], [9, 16], [11], [12], [13], [14])
groupBackward = Group(*groupTuple)

# Split-and-merge group
groupTuple = ([1], [2], [3], [4], [5, 6, 10, 15], [7, 17, 18], [8], [9, 16], [11], [12], [13], [14])
groupSplitAndMerge = Group(*groupTuple)
groupedNames = data.getGroupedNames(groupSplitAndMerge)
for names in groupedNames:
    print names
"""
['Age']
['HoursPerWeek']
['TotalHours']
['APM']
['SelectByHotkeys', 'AssignToHotkeys', 'NumberOfPACs', 'WorkersMade']
['UniqueHotkeys', 'ComplexUnitsMade', 'ComplexAbilitiesUsed']
['MinimapAttacks']
['MinimapRightClicks', 'UniqueUnitsMade']
['GapBetweenPACs']
['ActionLatency']
['ActionsInPAC']
['TotalMapExplored']
"""

# Compare with grouped strcutrue in the paper
import time

from okgtreg.Parameters import Parameters
from okgtreg.OKGTReg import OKGTReg

# print data
groupPaper = Group([1], [2], [3], [4], [5,6,7], [8,9], [10,11,12,13], [14], [15,16,17], [18])
kernel = Kernel('gaussian', sigma=0.5)
parameters = Parameters(groupPaper, kernel, [kernel]*groupPaper.size)
okgt = OKGTReg(data, parameters)

start_time = time.time()
fit = okgt.train('nystroem', 1000, 25)
elapsed_time = time.time() - start_time
print "R2 = ", fit['r2']
print "Elapsed time = ", elapsed_time
"""
With a large sample size, the number of ranks for Nystroem should also increase.

The preliminary experiment shows that:

rank = 10, R2 = 0.4184
rank = 100, R2 = 0.5672
rank = 300, R2 = 0.6613
rank = 500, R2 = 0.7021
rank = 1000, R2 = 0.7912
"""

# TODO: Other low rank approximation that keeps the number of retained ranks stable?
"""
Philosophically, this should depend on the underlying mechanism that generates the data.
If the data is essentially concenterated in a low dimensional space, then the number of the
retained ranks should be somewhat independent of the sample size.

However, if the data is not inherently lives in a low dimensional space. Then having more
data would reveal more and more variation directions. Thus the number of the retained ranks
should increase as more data are observed.
"""