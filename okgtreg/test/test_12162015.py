"""
Test the updated version of RandomGroup
"""
from okgtreg.Group import RandomGroup

# Given size and covariate indices
randomGroup1 = RandomGroup(4, [1,2,3,4,5,6], seed=25)
randomGroup1

# Given size and covariate indices (not start from 1 and randomly ordered)
randomGroup2 = RandomGroup(4, [2,4,3,7,6,10], seed=25)
randomGroup2

# Given size and total number of covariates
randomGroup3 = RandomGroup(4, nCovariates=10, seed=25)
randomGroup3

# Given size and named covariates (names are alphabetically ordered)
randomGroup4 = RandomGroup(4, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
randomGroup4

# Given size and named covariates (names are randomly ordered)
randomGroup5 = RandomGroup(4, ['BA', 'd', 'afg', 'aam', 'zfd', 'ghs'])
randomGroup5

# Size > number of covariates
randomGroup6 = RandomGroup(10, [1,2,3,4,5,6])

"""
Modified the interface of Data.getGroupedData by allow the input to be a list
"""
from okgtreg.DataSimulator import DataSimulator

n = 500
data, trueGroup = DataSimulator.SimData_Wang04WithInteraction(n)

# get a subset of the data corresponding to the group structure given as a list
groupList = [[1], [6,7]]
subData, subGroup = data.getGroupedData(groupList)
subData
data


"""
Modified the interface of Data.__getitem__ by allow different data types as key
"""
import string

from okgtreg.DataSimulator import DataSimulator

n = 500
data, trueGroup = DataSimulator.SimData_Wang04WithInteraction(n)

# string, when no names assigned in data
data['x1']

# string
data.setXNames(list(string.ascii_lowercase)[:7])
data.setYName('y')

data['b']  # successful
data['x1']  # no such name

# integer
data[1]  # covariate, success
data[0]  # response, success
data[100]  # index out of bound

# list of strings
data[['a', 'c']]  # success
data[['a', 'pan']]

# list of integers
data[[1,2,3]]  # success
data[[-1, 0, 2]]  # -1 out of bounds
