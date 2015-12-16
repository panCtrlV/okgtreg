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