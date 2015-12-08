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

"""
Test the new wrapper method Group.split()
"""
group = Group([1], [2,3,4], [5,6], [7])

# random split: no need to split
group.split(1, True)
# random split: split one covariate
for i in xrange(10):
    print group.split(2, True)
# random split: split one covariate with seeding
for i in xrange(10):
    print group.split(2, True, 1)
# complete split: no need to split
group.split(1)
group.split(4)
# complete split: split
group.split(2)
group.split(3)