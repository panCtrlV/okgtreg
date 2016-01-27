__author__ = 'panc'

'''
Reporting the simulation results for group structure detection
by backward partition method with penalty. Recall the penalty
is in the form of $\sum_\ell p_\ell^{p_\ell}$ where $p_\ell$ is
the number of variable for the $\ell$-th group in the structure.

# Data normalization: No
# Simulation time: 2016-01-25 12:05:05
# Kernel: Gaussian (0.5)
# Training method: vanilla
'''

from okgtreg.simulation.sim_01252016_1118.helper import *

# Directory path is same for all pickle files
dirname = 'okgtreg/simulation/sim_01252016_1118/'

'''
# Model 1 results
# ===============
# True model: Group([1], [2], [3], [4], [5], [6])
'''
filename = 'script-model-1.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 42 most frequent groupings ===
1 : (1,) : 14
2 : (2, 3) : 13
3 : (2, 4) : 13
4 : (1, 6) : 12
5 : (1, 2) : 12
6 : (4, 6) : 12
7 : (1, 5) : 11
8 : (5,) : 11
9 : (4, 5) : 11
10 : (2,) : 10

=== Top 10 out of 57 most frequent group structures ===
1 : ([1, 5], [2, 4], [3, 6]) : 6
2 : ([1, 3, 4], [2, 5, 6]) : 6
3 : ([1, 3, 6], [2, 4, 5]) : 5
4 : ([1, 6], [2, 3], [4, 5]) : 4
5 : ([1, 4, 6], [2, 3, 5]) : 4
6 : ([1, 2, 6], [3, 4, 5]) : 4
7 : ([1, 4, 5], [2, 3, 6]) : 4
8 : ([1, 2], [3, 5], [4, 6]) : 4
9 : ([1, 5], [2, 3], [4, 6]) : 3
10 : ([1, 6], [2], [3, 4, 5]) : 3
'''

'''
# Model 2 results
# ===============
# True model: Group([1], [2], [3], [4], [5], [6])
'''
filename = 'script-model-2.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 57 most frequent groupings ===
1 : (1,) : 10
2 : (1, 6) : 9
3 : (2,) : 8
4 : (1, 2, 4, 6) : 7
5 : (2, 5) : 7
6 : (2, 3, 4, 5, 6) : 7
7 : (1, 2) : 7
8 : (2, 6) : 7
9 : (3, 5) : 7
10 : (3,) : 6

=== Top 10 out of 46 most frequent group structures ===
1 : ([1], [2, 3, 4, 5, 6]) : 7
2 : ([1, 2, 4, 6], [3, 5]) : 6
3 : ([1, 2], [3, 4, 5, 6]) : 6
4 : ([1, 3, 4, 6], [2, 5]) : 5
5 : ([1, 3, 4, 5], [2, 6]) : 5
6 : ([1, 3], [2, 4, 5, 6]) : 4
7 : ([1, 6], [2, 3, 4, 5]) : 4
8 : ([1, 4], [2, 3, 5, 6]) : 3
9 : ([1, 3, 4, 5, 6], [2]) : 3
10 : ([1, 2, 3, 4, 6], [5]) : 3
'''

'''
# Model 3 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
filename = 'script-model-3.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 47 most frequent groupings ===
1 : (2, 3) : 22
2 : (4, 5, 6) : 21
3 : (1, 4, 5, 6) : 15
4 : (1, 3) : 14
5 : (1, 2) : 14
6 : (1,) : 13
7 : (2, 4, 5, 6) : 12
8 : (3, 4, 5, 6) : 12
9 : (1, 2, 3) : 11
10 : (3,) : 10

=== Top 10 out of 30 most frequent group structures ===
1 : ([1, 4, 5, 6], [2, 3]) : 15
2 : ([1, 3], [2, 4, 5, 6]) : 12
3 : ([1, 2], [3, 4, 5, 6]) : 11
4 : ([1, 2, 3], [4, 5, 6]) : 11
5 : ([1], [2, 3, 4, 5, 6]) : 6
6 : ([1], [2, 3], [4, 5, 6]) : 5
7 : ([1, 2, 4, 5, 6], [3]) : 5
8 : ([1, 3, 4, 5, 6], [2]) : 4
9 : ([1, 2], [3], [4, 5, 6]) : 3
10 : ([1, 2, 4, 6], [3, 5]) : 2
'''

'''
# Model 4 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
filename = 'script-model-4.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 42 most frequent groupings ===
1 : (4, 5, 6) : 64
2 : (1, 2, 3) : 24
3 : (2,) : 23
4 : (1, 3) : 21
5 : (1,) : 18
6 : (2, 3) : 18
7 : (1, 2) : 17
8 : (3,) : 12
9 : (4, 5) : 5
10 : (4, 6) : 5

=== Top 10 out of 31 most frequent group structures ===
1 : ([1, 2, 3], [4, 5, 6]) : 24
2 : ([1, 3], [2], [4, 5, 6]) : 18
3 : ([1, 2], [3], [4, 5, 6]) : 11
4 : ([1], [2, 3], [4, 5, 6]) : 10
5 : ([1, 4, 5, 6], [2, 3]) : 4
6 : ([1, 2], [3, 4, 5, 6]) : 4
7 : ([1, 4], [2, 3, 5, 6]) : 2
8 : ([1, 3], [2, 4], [5, 6]) : 2
9 : ([1], [2], [3, 6], [4, 5]) : 2
10 : ([1, 3, 4], [2, 5, 6]) : 2

The grouping of [4,5,6] is correctly identified most often.
But the grouping of [1,2,3] is determined quite often.
'''

'''
# Model 5 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
filename = 'script-model-5.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 39 most frequent groupings ===
1 : (4, 5, 6) : 73
2 : (3,) : 31
3 : (1,) : 30
4 : (1, 2, 3) : 28
5 : (2,) : 27
6 : (2, 3) : 14
7 : (1, 2) : 13
8 : (1, 3) : 9
9 : (3, 5, 6) : 5
10 : (2, 3, 5) : 4

=== Top 10 out of 25 most frequent group structures ===
1 : ([1, 2, 3], [4, 5, 6]) : 28
2 : ([1], [2], [3], [4, 5, 6]) : 19
3 : ([1, 2], [3], [4, 5, 6]) : 10
4 : ([1], [2, 3], [4, 5, 6]) : 9
5 : ([1, 3], [2], [4, 5, 6]) : 7
6 : ([1, 4, 6], [2, 3, 5]) : 4
7 : ([1, 4, 5, 6], [2, 3]) : 3
8 : ([1, 2, 4], [3, 5, 6]) : 2
9 : ([1], [2, 4], [3, 5, 6]) : 2
10 : ([1, 2, 5], [3, 4, 6]) : 1


The grouping of [4,5,6] is identified successfully most
of time (73%). The second and third places on the top
grouping list are the univarate groups [3] and [1].

If we simply look at the top 3 groupings in this example,
we could identify the true group structure as ([1],[2],[3],[]4,5,6).

In the top group structure list, the true group structure
is identified second most often.
'''

'''
# Model 6 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
filename = 'script-model-6.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 42 most frequent groupings ===
1 : (1, 2, 3, 4, 6) : 14
2 : (5,) : 14
3 : (1,) : 11
4 : (2, 3, 4, 5, 6) : 11
5 : (1, 3, 4, 5, 6) : 9
6 : (2,) : 9
7 : (6,) : 8
8 : (1, 2, 3, 4, 5) : 8
9 : (3,) : 7
10 : (1, 4) : 7

=== Top 10 out of 21 most frequent group structures ===
1 : ([1, 2, 3, 4, 6], [5]) : 14
2 : ([1], [2, 3, 4, 5, 6]) : 11
3 : ([1, 3, 4, 5, 6], [2]) : 9
4 : ([1, 2, 3, 4, 5], [6]) : 8
5 : ([1, 4], [2, 3, 5, 6]) : 7
6 : ([1, 2, 4, 5, 6], [3]) : 7
7 : ([1, 2, 3, 5, 6], [4]) : 6
8 : ([1, 5], [2, 3, 4, 6]) : 5
9 : ([1, 2, 3, 5], [4, 6]) : 5
10 : ([1, 3, 4, 5], [2, 6]) : 4
'''

'''
# Model 7 results
# ===============
# True model: Group([1], [2], [3], [4, 5, 6])
'''
filename = 'script-model-7.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 50 most frequent groupings ===
1 : (1, 4, 5, 6) : 10
2 : (1, 2, 4, 6) : 9
3 : (1, 2) : 9
4 : (2, 3) : 9
5 : (3, 5) : 9
6 : (3, 4, 5, 6) : 9
7 : (1, 3) : 8
8 : (2, 4, 5, 6) : 8
9 : (1, 4) : 6
10 : (2, 3, 5, 6) : 6

=== Top 10 out of 26 most frequent group structures ===
1 : ([1, 4, 5, 6], [2, 3]) : 9
2 : ([1, 2, 4, 6], [3, 5]) : 9
3 : ([1, 2], [3, 4, 5, 6]) : 9
4 : ([1, 3], [2, 4, 5, 6]) : 8
5 : ([1, 4], [2, 3, 5, 6]) : 6
6 : ([1, 2, 3, 4], [5, 6]) : 5
7 : ([1, 5], [2, 3, 4, 6]) : 5
8 : ([1, 2, 3], [4, 5, 6]) : 5
9 : ([1, 3, 4, 6], [2, 5]) : 4
10 : ([1, 3, 4, 5], [2, 6]) : 4
'''

'''
# Model 8 results
# ===============
# True model: Group([1], [2], [3], [4], [5], [6])
'''
filename = 'script-model-8.pkl'
reportResults(dirname, filename)
'''
Still running ...
'''

'''
# Model 9 results
# ===============
# True model: Group([1], [2, 3], [4, 5, 6])
'''
filename = 'script-model-9.pkl'
reportResults(dirname, filename)
'''
=== Top 10 out of 19 most frequent groupings ===
1 : (2, 3) : 31
2 : (1, 5, 6) : 25
3 : (4, 5, 6) : 25
4 : (1, 4, 5) : 24
5 : (2, 3, 4) : 20
6 : (2, 3, 6) : 20
7 : (1, 4, 6) : 16
8 : (1, 2, 3) : 15
9 : (2, 3, 5) : 14
10 : (1,) : 11

=== Top 10 out of 15 most frequent group structures ===
1 : ([1, 5, 6], [2, 3, 4]) : 19
2 : ([1, 4, 5], [2, 3, 6]) : 18
3 : ([1, 2, 3], [4, 5, 6]) : 15
4 : ([1, 4, 6], [2, 3, 5]) : 12
5 : ([1], [2, 3], [4, 5, 6]) : 10
6 : ([1, 5, 6], [2, 3], [4]) : 6
7 : ([1, 4, 5], [2, 3], [6]) : 6
8 : ([1, 4, 6], [2, 3], [5]) : 4
9 : ([1, 6], [2, 3], [4, 5]) : 3
10 : ([1, 4], [2, 3], [5, 6]) : 2
'''

'''
# Model 10 results
# ===============
# True model: Group([1], [2, 3], [4, 5, 6])
'''
filename = 'script-model-10.pkl'
reportResults(dirname, filename)
'''
=== Top 7 out of 7 most frequent groupings ===
1 : (4, 5, 6) : 100
2 : (1,) : 96
3 : (2, 3) : 95
4 : (2,) : 4
5 : (1, 3) : 3
6 : (3,) : 2
7 : (1, 2) : 1

=== Top 4 out of 4 most frequent group structures ===
1 : ([1], [2, 3], [4, 5, 6]) : 95
2 : ([1, 3], [2], [4, 5, 6]) : 3
3 : ([1], [2], [3], [4, 5, 6]) : 1
4 : ([1, 2], [3], [4, 5, 6]) : 1

In this example, the backward successfully identified
the true group structure as the optimal one. The grouping
of the variables 4,5,6 are identified 100% of time.
'''
