__author__ = 'panc'

'''
Report the simulation results of group structure detection
using forward inclusion/selection with
'''

import pickle

from okgtreg.simulation.sim_01262016_1501.helper import *

# Directory path is same for all pickle files
dirname = 'okgtreg/simulation/sim_01262016_1501/'

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
groupingCounter = groupingFrequency(groupList)
print("=== Top 10 most frequent groupings ===")
for i in range(10):
    item = groupingCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent groupings ===
1 : (1, 3) : 10
2 : (2, 3, 5, 6) : 10
3 : (2, 4, 5, 6) : 10
4 : (1, 4) : 10
5 : (1, 2, 4, 6) : 9
6 : (1, 2) : 9
7 : (2, 3) : 9
8 : (3, 5) : 9
9 : (3, 4, 5, 6) : 9
10 : (1, 4, 5, 6) : 9
'''
groupCounter = groupFrequency(groupList)
print("=== Top 10 most frequent group structures ===")
for i in range(10):
    item = groupCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent group structures ===
1 : ((1, 3), (2, 4, 5, 6)) : 10
2 : ((2, 3, 5, 6), (1, 4)) : 10
3 : ((2, 3), (1, 4, 5, 6)) : 9
4 : ((1, 2), (3, 4, 5, 6)) : 9
5 : ((1, 2, 4, 6), (3, 5)) : 9
6 : ((1, 3, 4, 6), (2, 5)) : 8
7 : ((1, 3, 4, 5), (2, 6)) : 7
8 : ((2, 3, 4, 6), (1, 5)) : 7
9 : ((5, 6), (1, 2, 3, 4)) : 7
10 : ((2, 3, 4, 5), (1, 6)) : 5

The simulation results for normalized data is similar to
those without normalization.

The variables are partitioned into two groups, with 2-4
split in the top 10 group structures.

The top group structure ((1, 2, 4, 6), (3, 5)) identified
without normalization in "sim_01252016_1017" is now in the
5-th place. The top group structure ((1, 3), (2, 4, 5, 6))
was not listed as one of the top 10 in "sim_01252016_1017".

Another observation is that, the variability of group structure
estimates is smaller when normalization is used. When normalization
is used, 94/100 estimates belong to the top 10. The same
 measure is only 49/100.
'''

'''
# Model 2 results
# ===============
# True model: Group([1], [2], [3], [4], [5], [6])
'''
# unpickle results file
filename = 'script-model-2.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = groupingFrequency(groupList)
print("=== Top 10 most frequent groupings ===")
for i in range(10):
    item = groupingCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent groupings ===
1 : (1, 2, 4, 6) : 8
2 : (2, 5) : 8
3 : (1, 4) : 8
4 : (2, 3, 5, 6) : 8
5 : (3, 5) : 8
6 : (1, 3, 4, 6) : 8
7 : (5, 6) : 7
8 : (1, 2) : 7
9 : (2, 3) : 7
10 : (1, 2, 3, 4) : 7
'''
groupCounter = groupFrequency(groupList)
print("=== Top 10 most frequent group structures ===")
for i in range(10):
    item = groupCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent group structures ===
1 : ((1, 3, 4, 6), (2, 5)) : 8
2 : ((1, 2, 4, 6), (3, 5)) : 8
3 : ((2, 3, 5, 6), (1, 4)) : 8
4 : ((2, 3), (1, 4, 5, 6)) : 7
5 : ((1, 2), (3, 4, 5, 6)) : 7
6 : ((5, 6), (1, 2, 3, 4)) : 7
7 : ((1, 3), (2, 4, 5, 6)) : 6
8 : ((2, 3, 4, 6), (1, 5)) : 6
9 : ((1, 3, 4, 5), (2, 6)) : 5
10 : ((3,), (1, 2, 4, 5, 6)) : 4
'''

'''
# Model 3 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-3.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = groupingFrequency(groupList)
print("=== Top 10 most frequent groupings ===")
for i in range(10):
    item = groupingCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent groupings ===
1 : (1, 2) : 23
2 : (2, 3) : 23
3 : (3, 4, 5, 6) : 23
4 : (1, 4, 5, 6) : 23
5 : (1, 3) : 21
6 : (2, 4, 5, 6) : 21
7 : (1, 2, 4, 6) : 4
8 : (1, 4) : 4
9 : (2, 3, 5, 6) : 4
10 : (3, 5) : 4
'''
groupCounter = groupFrequency(groupList)
print("=== Top 10 most frequent group structures ===")
for i in range(10):
    item = groupCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent group structures ===
1 : ([1, 4, 5, 6], [2, 3]) : 23
2 : ([1, 2], [3, 4, 5, 6]) : 23
3 : ([1, 3], [2, 4, 5, 6]) : 21
4 : ([1, 4], [2, 3, 5, 6]) : 4
5 : ([1, 2, 4, 6], [3, 5]) : 4
6 : ([1, 6], [2, 3, 4, 5]) : 3
7 : ([1, 3, 4, 5], [2, 6]) : 3
8 : ([1, 2, 3], [4, 5, 6]) : 3
9 : ([1, 2, 3, 4, 6], [5]) : 2
10 : ([1, 3, 4, 5, 6], [2]) : 2
'''

'''
# Model 4 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-4.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = groupingFrequency(groupList)
print("=== Top 10 most frequent groupings ===")
for i in range(10):
    item = groupingCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent groupings ===
1 : (1, 2) : 18
2 : (3, 4, 5, 6) : 18
3 : (2, 3) : 12
4 : (1, 4, 5, 6) : 12
5 : (1, 3) : 11
6 : (2, 4, 5, 6) : 11
7 : (1, 2, 4, 6) : 7
8 : (1, 4) : 7
9 : (2, 3, 5, 6) : 7
10 : (3, 5) : 7
'''
groupCounter = groupFrequency(groupList)
print("=== Top 10 most frequent group structures ===")
for i in range(10):
    item = groupCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent group structures ===
1 : ([1, 2], [3, 4, 5, 6]) : 18
2 : ([1, 4, 5, 6], [2, 3]) : 12
3 : ([1, 3], [2, 4, 5, 6]) : 11
4 : ([1, 4], [2, 3, 5, 6]) : 7
5 : ([1, 2, 4, 6], [3, 5]) : 7
6 : ([1, 3, 4, 5], [2, 6]) : 6
7 : ([1, 5], [2, 3, 4, 6]) : 5
8 : ([1, 3, 4, 6], [2, 5]) : 5
9 : ([1, 6], [2, 3, 4, 5]) : 5
10 : ([1, 2, 5, 6], [3, 4]) : 4
'''


'''
# Model 5 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-5.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = groupingFrequency(groupList)
print("=== Top 10 most frequent groupings ===")
for i in range(10):
    item = groupingCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent groupings ===
1 : (4, 5, 6) : 21
2 : (5, 6) : 12
3 : (1, 2, 3, 4) : 12
4 : (1, 2) : 11
5 : (1,) : 11
6 : (2,) : 11
7 : (1, 3) : 9
8 : (3,) : 8
9 : (1, 2, 3, 5) : 8
10 : (2, 3) : 8
'''
groupCounter = groupFrequency(groupList)
print("=== Top 10 most frequent group structures ===")
for i in range(10):
    item = groupCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent group structures ===
1 : ([1, 2, 3, 4], [5, 6]) : 12
2 : ([1, 2, 3, 5], [4, 6]) : 8
3 : ([1, 2], [3, 4, 5, 6]) : 8
4 : ([1, 2, 3, 6], [4, 5]) : 7
5 : ([1, 3], [2, 4, 5, 6]) : 6
6 : ([1, 4], [2, 3, 5, 6]) : 5
7 : ([1, 3, 4, 5], [2, 6]) : 5
8 : ([1, 3, 4, 6], [2, 5]) : 4
9 : ([1, 4, 5, 6], [2, 3]) : 4
10 : ([1], [2], [3], [4, 5, 6]) : 3

The true group structure is on the bottom of the
top 10 group structures.
'''

'''
# Model 6 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-6.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = groupingFrequency(groupList)
print("=== Top 10 most frequent groupings ===")
for i in range(10):
    item = groupingCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent groupings ===
1 : (1, 2, 3, 4, 6) : 16
2 : (5,) : 16
3 : (1,) : 13
4 : (2, 3, 4, 5, 6) : 13
5 : (1, 2, 3, 4, 5) : 12
6 : (6,) : 12
7 : (1, 3, 4, 5, 6) : 10
8 : (1, 2, 3, 5, 6) : 10
9 : (2,) : 10
10 : (4,) : 10
'''
groupCounter = groupFrequency(groupList)
print("=== Top 10 most frequent group structures ===")
for i in range(10):
    item = groupCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent group structures ===
1 : ([1, 2, 3, 4, 6], [5]) : 16
2 : ([1], [2, 3, 4, 5, 6]) : 13
3 : ([1, 2, 3, 4, 5], [6]) : 12
4 : ([1, 3, 4, 5, 6], [2]) : 10
5 : ([1, 2, 3, 5, 6], [4]) : 10
6 : ([1, 2, 4, 5, 6], [3]) : 8
7 : ([1, 2, 3, 5], [4, 6]) : 5
8 : ([1, 3, 4, 5], [2, 6]) : 5
9 : ([1, 2, 3, 6], [4, 5]) : 3
10 : ([1, 2, 5, 6], [3, 4]) : 3
'''

'''
# Model 7 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
# unpickle results file
filename = 'script-model-7.pkl'
with open(dirname + filename, 'rb') as f:
    res = pickle.load(f)

groupList = res.keys()
groupingCounter = groupingFrequency(groupList)
print("=== Top 10 most frequent groupings ===")
for i in range(10):
    item = groupingCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent groupings ===
1 : (1, 4) : 10
2 : (2, 3, 5, 6) : 10
3 : (1, 2, 4, 6) : 9
4 : (3, 5) : 9
5 : (5, 6) : 8
6 : (2, 5) : 8
7 : (1, 2, 3, 4) : 8
8 : (1, 3, 4, 6) : 8
9 : (1, 2) : 7
10 : (3, 4, 5, 6) : 7
'''
groupCounter = groupFrequency(groupList)
print("=== Top 10 most frequent group structures ===")
for i in range(10):
    item = groupCounter[i]
    print("%d : %s : %d" % (i + 1, item[0], item[1]))
'''
=== Top 10 most frequent group structures ===
1 : ([1, 4], [2, 3, 5, 6]) : 10
2 : ([1, 2, 4, 6], [3, 5]) : 9
3 : ([1, 3, 4, 6], [2, 5]) : 8
4 : ([1, 2, 3, 4], [5, 6]) : 8
5 : ([1, 2], [3, 4, 5, 6]) : 7
6 : ([1, 3], [2, 4, 5, 6]) : 6
7 : ([1, 3, 4, 5], [2, 6]) : 6
8 : ([1, 2, 3, 5], [4, 6]) : 6
9 : ([1, 5], [2, 3, 4, 6]) : 6
10 : ([1, 2, 3, 6], [4, 5]) : 5
'''

'''
# Model 8 results
# ===============
# True model: Group([1], [2], [3], [4], [5], [6])
'''
filename = 'script-model-8.pkl'
reportResults(dirname, filename)
'''
=== Top 10 most frequent groupings ===
1 : (5,) : 57
2 : (4,) : 40
3 : (3, 6) : 36
4 : (1,) : 31
5 : (2, 3, 6) : 29
6 : (1, 4) : 21
7 : (6,) : 21
8 : (1, 5) : 18
9 : (2,) : 17
10 : (2, 4) : 16

=== Top 10 most frequent group structures ===
1 : ([1, 4], [2, 3, 6], [5]) : 15
2 : ([1], [2, 4], [3, 6], [5]) : 7
3 : ([1, 2], [3, 6], [4], [5]) : 7
4 : ([1, 5], [2, 4], [3, 6]) : 5
5 : ([1, 5], [2, 3, 6], [4]) : 5
6 : ([1], [2, 3], [4], [5], [6]) : 5
7 : ([1], [2, 3, 6], [4, 5]) : 3
8 : ([1], [2], [3, 6], [4]) : 3
9 : ([1], [2], [3, 6], [4], [5]) : 3
10 : ([1, 2], [3, 4, 6], [5]) : 2
'''

'''
# Model 9 results
# ===============
# True model: Group([1], [2, 3], [4, 5, 6])
'''
filename = 'script-model-9.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 10 most frequent groupings ===
1 : (4, 5) : 38
2 : (1, 2, 3, 6) : 38
3 : (1, 2, 3, 4) : 29
4 : (5, 6) : 29
5 : (1, 2, 3, 5) : 25
6 : (4, 6) : 25
7 : (2, 3, 4, 6) : 5
8 : (1, 5) : 5
9 : (1, 4) : 3
10 : (2, 3, 5, 6) : 3

=== Top 5 out of 5 most frequent group structures ===
1 : ([1, 2, 3, 6], [4, 5]) : 38
2 : ([1, 2, 3, 4], [5, 6]) : 29
3 : ([1, 2, 3, 5], [4, 6]) : 25
4 : ([1, 5], [2, 3, 4, 6]) : 5
5 : ([1, 4], [2, 3, 5, 6]) : 3
'''

'''
# Model 10 results
# ===============
# True model: Group([1], [2, 3], [4, 5, 6])
'''
filename = 'script-model-10.pkl'
reportResults(dirname, filename)
'''
=== Top 6 out of 6 most frequent groupings ===
1 : (4, 5, 6) : 99
2 : (1, 2, 3) : 96
3 : (1, 3) : 3
4 : (2,) : 3
5 : (2, 3) : 1
6 : (1, 4, 5, 6) : 1

=== Top 3 out of 3 most frequent group structures ===
1 : ([1, 2, 3], [4, 5, 6]) : 96
2 : ([1, 3], [2], [4, 5, 6]) : 3
3 : ([1, 4, 5, 6], [2, 3]) : 1


The true grouped is not detected in the simulation after
data being normalized. In "sim_10262016_1501", the true
group structure ([1], [2, 3], [4, 5, 6]) is identified as
the top one.
'''
