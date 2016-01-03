__author__ = 'panc'

"""
Application of forward procedure on SkillCraft1 data set.

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
print data
