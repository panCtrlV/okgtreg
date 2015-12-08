"""
Test the new method Group._randomSplitOneGroup()
"""

from okgtreg.Group import Group

group = Group([1], [2,3,4], [5,6], [7])

# no need to split
group._randomSplitOneGroup(1)
# random split one covariate
group._randomSplitOneGroup(2)
# random split with seeding
group._randomSplitOneGroup(2, 25)