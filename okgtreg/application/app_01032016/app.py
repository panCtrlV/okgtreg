__author__ = 'panc'

"""
1. Application of forward procedure on SkillCraft1 data set.

The data are normalized before applying forward procedure.
"""
from sklearn import preprocessing
from okgtreg import *
from okgtreg.groupStructureDetection import *

data = readSkillCraft1()
kernel = Kernel('gaussian', sigma=0.5)

# Normalize data
data.y = preprocessing.scale(data.y)
data.X = preprocessing.scale(data.X)

res = forwardSelection(data, kernel, 'nystroem', seed=25)
"""
SELECTED GROUP STRUCTURE:

    ([1], [2], [3, 8], [4, 18], [5], [6, 16], [7, 9], [10, 15], [11], [12], [13], [14], [17])
"""

"""
2. Reduce the data to include only 15 game related attributes.
   Fit OKGT on the reduced data with the fully additive structure.
"""
reduceData = Data(data.y, data.X[:, 3:])
reduceData.xnames = data.xnames[3:]

additiveGroup = Group(*tuple([[i + 1] for i in range(reduceData.p)]))
okgt = OKGTReg(reduceData, kernel=kernel, group=additiveGroup)
fit = okgt.train('nystroem', 200, 25)
print fit['r2']
"""
Out[6]: 0.6260017046656651
"""

import matplotlib.pyplot as plt

plt.scatter(data.y, fit['g'])
plt.title("LeagueIndex")

j = 4;
plt.scatter(data.X[:, j], fit['f'][:, j])

"""
3. Grouping all attributes in a single group, then fit OKGT.
"""
singleGroup = Group([i + 1 for i in range(reduceData.p)])
okgt2 = OKGTReg(reduceData, kernel=kernel, group=singleGroup)
fit2 = okgt2.train('nystroem', 200, 25)
print fit2['r2']  # 0.098430952157

"""
4. By grouping related game attributes together, an OKGT is fitted again
   on the reduced data set.
"""
print reduceData
categorizedGroup = Group([1], [2, 3, 4], [5, 6], [7, 8, 9, 10], [11], [12, 13, 14], [15])
okgt3 = OKGTReg(reduceData, kernel=kernel, group=categorizedGroup)
fit3 = okgt3.train('nystroem', 200, 25)
print fit3['r2']  # 0.570957347493
