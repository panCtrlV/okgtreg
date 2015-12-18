"""
Test the updated split method in Group, which can randomly split one
or multiple covariates from a group.
"""

from okgtreg.Group import Group


group = Group([1], [2,3,4,5,6,7], [8,9,10])

# Complete split one covariate
print group.split(2)

# Randomly split one covariate
print group.split(2, True, 25)
print group.split(2, True, 26)

# Randomly split multiple covariates
print group.split(2, True, 25, 2)
print group.split(2, True, 25, 3)
print group.split(2, True, 25, 4)
print group.split(2, True, 25, 5)
print group.split(2, True, 25, 6)

