__author__ = 'panc'

'''
Reporting the simulation results of group structure detection
by forward inclusion/selection with capacity penalty. Recall
that the selection criteria is to minimize the penalized $R^2$
where the penalty is in the form of $\lambda p_{\ell}^{p_{\ell}}$.
Here we use $\lambda = 1e-5$ for all models.
'''

import pickle

from okgtreg.simulation.sim_01252016_1017.helper import *

# Directory path is same for all pickle files
dirname = 'okgtreg/simulation/sim_01252016_1017/'

'''
# Model 1 results
# ===============
# True model: Group([1], [2], [3], [4], [5], [6])
'''
# unpickle results file
filename = 'script-model-1.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(1, 2, 4, 6) : 8
(1,) : 8
(3,) : 8
(2, 3) : 8
(3, 5) : 8
(1, 4, 5, 6) : 8
(5,) : 7
(2, 4, 5, 6) : 7
(1, 2) : 7
(1, 6) : 6

Though the true group structure is fully additive,
the simulation results tends to group multiple covariates
together.
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 2, 4, 6], [3, 5]) : 8
([1, 4, 5, 6], [2, 3]) : 7
([1, 2], [3, 4, 5, 6]) : 5
([1, 3, 4], [2, 5, 6]) : 5
([1], [2, 4, 5, 6], [3]) : 4
([1, 2, 3, 5], [4, 6]) : 4
([1, 6], [2, 3, 4, 5]) : 4
([1, 3, 4, 6], [2, 5]) : 4
([1, 3, 4, 5], [2, 6]) : 4
([1, 5], [2, 3, 4, 6]) : 4

The simulation results show the algorithm tends to pick
a group structure with two groups in a 4-2 way. That is
one group with 4 variables and the other with 2 variables.
'''

'''
# Model 2 results
# ===============
Group([1], [2], [3], [4], [5], [6])
'''
# unpickle results file
filename = 'script-model-2.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(1,) : 10
(1, 2, 4, 6) : 8
(1, 2) : 8
(3,) : 7
(5,) : 7
(2, 3, 4, 5, 6) : 7
(3, 5) : 7
(3, 4, 5, 6) : 7
(2, 5) : 6
(1, 3, 4, 5) : 6
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 2], [3, 4, 5, 6]) : 7
([1], [2, 3, 4, 5, 6]) : 7
([1, 2, 4, 6], [3, 5]) : 7
([1, 3, 4, 6], [2, 5]) : 6
([1, 3, 4, 5], [2, 6]) : 6
([1, 4], [2, 3, 5, 6]) : 5
([1, 2, 3, 4], [5, 6]) : 5
([1, 4, 5, 6], [2, 3]) : 4
([1, 3, 5, 6], [2, 4]) : 4
([1, 5], [2, 3, 4, 6]) : 4

Similar result as that of model 1.
'''

'''
# Model 3 results
# ===============
Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-3.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(3, 4, 5, 6) : 19
(1, 2) : 18
(1, 3) : 17
(2, 4, 5, 6) : 16
(2, 3) : 11
(2,) : 11
(1,) : 10
(1, 4, 5, 6) : 10
(2, 3, 4, 5, 6) : 7
(4, 5, 6) : 7

Variables 4, 5, 6 tend to be included in one group,
which is conformable to the true group structure.
However, other variables are also determined to be
in the same group. Sometimes, the other variables
tend to be grouped among themselves.
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 2], [3, 4, 5, 6]) : 17
([1, 3], [2, 4, 5, 6]) : 16
([1, 4, 5, 6], [2, 3]) : 10
([1], [2, 3, 4, 5, 6]) : 7
([1, 3, 4, 5, 6], [2]) : 5
([1, 2, 3], [4, 5, 6]) : 5
([1, 2, 4, 6], [3, 5]) : 4
([1, 2, 4, 5, 6], [3]) : 3
([1, 6], [2, 3, 4, 5]) : 3
([1, 2, 3, 4, 6], [5]) : 3

The algorithm successfully detects that 4,5,6 belong
to the same group most of time. However, the partitions
of the other three variables are not successful. While,
one is picked as a univariate group, the other two are
coffined with [4,5,6] as a single group.
'''


'''
# Model 4 results
# ===============
Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-4.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(1, 2) : 16
(3, 4, 5, 6) : 15
(2, 3) : 14
(1, 4, 5, 6) : 13
(1, 3) : 10
(2, 4, 5, 6) : 9
(1,) : 8
(3,) : 7
(3, 5) : 7
(2,) : 7
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 2], [3, 4, 5, 6]) : 15
([1, 4, 5, 6], [2, 3]) : 11
([1, 3], [2, 4, 5, 6]) : 6
([1, 6], [2, 3, 4, 5]) : 5
([1, 2, 4, 6], [3, 5]) : 5
([1, 5], [2, 3, 4, 6]) : 5
([1, 4], [2, 3, 5, 6]) : 5
([1, 2, 3, 6], [4, 5]) : 4
([1], [2, 4, 5, 6], [3]) : 3
([1, 3, 4, 5], [2, 6]) : 3
'''

'''
# Model 5 results
# ===============
Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-5.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(4, 5, 6) : 28
(1,) : 17
(2,) : 16
(3,) : 15
(1, 3) : 10
(1, 2) : 10
(1, 2, 3) : 9
(2, 4, 5, 6) : 7
(3, 4, 5, 6) : 7
(5, 6) : 5

The grouping frequency is conformable to the true group
structure. That is that variables 4,5,6 are together, and
1, 2, 3 each forms a univariate group.
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 2, 3], [4, 5, 6]) : 7
([1, 2], [3, 4, 5, 6]) : 6
([1, 3], [2, 4, 5, 6]) : 6
([1], [2], [3], [4, 5, 6]) : 5
([1, 2, 3, 4], [5, 6]) : 5
([1, 2, 3, 5], [4, 6]) : 4
([1, 3, 4, 6], [2, 5]) : 4
([1, 2, 4, 6], [3, 5]) : 4
([1, 2], [3], [4, 5, 6]) : 4
([1, 4, 5, 6], [2, 3]) : 3

The true group structure is correctly identified 4/100 times,
though most of time two-group structures are determined
to be optimal.
'''

'''
# Model 6 results
# ===============
Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-6.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(1, 2, 3, 4, 6) : 17
(5,) : 17
(1,) : 12
(2, 3, 4, 5, 6) : 12
(3,) : 10
(1, 2, 4, 5, 6) : 10
(1, 3, 4, 5, 6) : 9
(1, 2, 3, 5, 6) : 9
(1, 2, 3, 4, 5) : 9
(2,) : 9
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 2, 3, 4, 6], [5]) : 17
([1], [2, 3, 4, 5, 6]) : 12
([1, 2, 4, 5, 6], [3]) : 10
([1, 3, 4, 5, 6], [2]) : 9
([1, 2, 3, 4, 5], [6]) : 9
([1, 2, 3, 5, 6], [4]) : 9
([1, 3, 4, 5], [2, 6]) : 6
([1, 2, 3, 5], [4, 6]) : 5
([1, 5], [2, 3, 4, 6]) : 5
([1, 4], [2, 3, 5, 6]) : 4
'''

'''
# Model 7 results
# ===============
Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-7.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(1, 2, 4, 6) : 9
(3, 5) : 9
(6,) : 8
(1, 2, 3, 4, 5) : 8
(1, 4) : 7
(2, 3, 5, 6) : 7
(5, 6) : 6
(2, 5) : 6
(2, 3, 4, 6) : 6
(3,) : 6
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 2, 4, 6], [3, 5]) : 9
([1, 2, 3, 4, 5], [6]) : 8
([1, 4], [2, 3, 5, 6]) : 7
([1, 3, 4, 5, 6], [2]) : 6
([1, 2, 4, 5, 6], [3]) : 6
([1, 3, 4, 6], [2, 5]) : 6
([1, 5], [2, 3, 4, 6]) : 6
([1, 2, 3, 4], [5, 6]) : 6
([1, 2], [3, 4, 5, 6]) : 5
([1, 3], [2, 4, 5, 6]) : 5
'''

'''
# Model 8 results
# ===============
Group([1], [2], [3], [4], [5], [6])
'''
# unpickle results file
filename = 'script-model-8.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(5,) : 42
(3, 6) : 27
(4,) : 27
(2, 3, 6) : 24
(6,) : 19
(1,) : 18
(1, 4) : 17
(2, 3) : 13
(2, 4) : 13
(1, 2) : 12
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 4], [2, 3, 6], [5]) : 10
([1, 3, 6], [2, 4], [5]) : 4
([1, 5], [2, 3, 6], [4]) : 4
([1, 4], [2, 3, 6]) : 4
([3, 6],) : 4
([1], [2, 3], [4], [5], [6]) : 4
([3], [6]) : 3
([1, 2], [3, 6], [4]) : 3
([1], [2, 4], [3, 6], [5]) : 3
([1, 5], [2, 3, 6]) : 3
'''

'''
# Model 9 results
# ===============
Group([1], [2, 3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-9.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(1, 2, 3) : 58
(4, 5, 6) : 51
(1, 5, 6) : 15
(2, 3) : 14
(2, 3, 5) : 11
(1, 4, 5) : 10
(4,) : 10
(2, 3, 6) : 9
(6,) : 8
(1, 4, 6) : 8
'''
groupCounter = printGroupFrequency(groupList)
'''
([1, 2, 3], [4, 5, 6]) : 49
([1, 5, 6], [2, 3], [4]) : 8
([1, 4, 5], [2, 3, 6]) : 8
([1, 5, 6], [2, 3, 4]) : 7
([1, 4, 6], [2, 3, 5]) : 7
([1, 2, 3], [4, 6], [5]) : 5
([1, 2, 3], [4, 5], [6]) : 3
([1, 4, 5], [2, 3], [6]) : 2
([1, 4], [2, 3, 5], [6]) : 2
([1], [2, 3], [4, 5, 6]) : 2
'''

'''
# Model 10 results
# ===============
Group([1], [2, 3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-10.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = printGroupingFrequency(groupList)
'''
(4, 5, 6) : 100
(1,) : 53
(2, 3) : 52
(2,) : 46
(1, 3) : 45
(3,) : 2
(1, 2) : 1
(1, 2, 3) : 1
'''
groupCounter = printGroupFrequency(groupList)
'''
([1], [2, 3], [4, 5, 6]) : 52
([1, 3], [2], [4, 5, 6]) : 45
([1], [2], [3], [4, 5, 6]) : 1
([1, 2], [3], [4, 5, 6]) : 1
([1, 2, 3], [4, 5, 6]) : 1
'''
