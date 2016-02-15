__author__ = 'panc'

from okgtreg import *

g1 = Group([1, 2], [3, 4], [5, 6])
g2 = Group([1, 2, 3, 4], [5, 6])

g1 < g2  # True
g2 < g1  # False

g3 = Group([1, 2, 3], [4, 5, 6])

g1 < g3  # False
g3 < g1  # False
g2 < g3  # False
g3 < g2  # False

g4 = Group([1, 2, 3], [4, 5])

g4 < g3  # ValueError, different number of covariates

g5 = Group([1, 2], [5, 6], [3, 4])
g5 <= g1  # True
g1 <= g5  # True
g1 < g5  # False
g5 < g1  # False

# All possible group structures for the
# covariates in the current group structure
g1.allGroupStructures()

# All amiable group structures
g1.amiableGroupStructures()
Group(group_struct_string="[[1], [2,3], [4], [5]]").amiableGroupStructures()
