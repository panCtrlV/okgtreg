from core.Group import *

group = Group([1,2], [3], [4,5,6])
group.partition
newGroup = group.addNewCovariateAsGroup(7)
newGroup.partition
newGroup.p

group = Group([1,2])
group.partition
newGroup = group.addNewCovariateToGroup(3, 1)
newGroup.partition
newGroup = group.addNewCovariateToGroup(4, 2)
newGroup = group.addNewCovariateToGroup(1, 1)


