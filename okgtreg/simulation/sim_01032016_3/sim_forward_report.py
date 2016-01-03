__author__ = 'panc'

import pickle
from okgtreg.simulation.sim_01032016_3.utility import *

pklfile = open('okgtreg/simulation/sim_01032016_3/sim_forward.py.pkl', 'rb')
groups, r2s = pickle.load(pklfile)
pklfile.close()

np.mean(r2s)

printGroupingFrequency(groups)
"""
The following grouping frequencies show that the procedure
successfully recovers the bi-variate groupings as the most
common structures in the simulation.

(1, 2) : 97
(5, 6) : 96
(7, 8) : 92
(3, 4) : 85
(10,) : 55
(9,) : 52
(9, 10) : 45
(4,) : 13
(3,) : 11
(8,) : 7
(7,) : 6
(6,) : 3
(3, 5) : 2
(2,) : 2
(1, 3) : 1
(5, 7, 8) : 1
(4, 9) : 1
(1,) : 1
(5,) : 1
(3, 9) : 1
(1, 9) : 1
(2, 7) : 1
(4, 6) : 1
"""

printGroupFrequency(groups)
"""
The following group structure frequencies show that the true structure
is detected as the second most frequent in the simulation. Though the
most frequent structure is different from the true model, they are different
only in the last group. If this is the structure detected for a data set, it
should give us enough guidance about the data structure for further exploration.

([1, 2], [3, 4], [5, 6], [7, 8], [9], [10]) : 39
([1, 2], [3, 4], [5, 6], [7, 8], [9, 10]) : 38
([1, 2], [3], [4], [5, 6], [7, 8], [9], [10]) : 6
([1, 2], [3], [4], [5, 6], [7, 8], [9, 10]) : 4
([1, 2], [3, 4], [5, 6], [7], [8], [9], [10]) : 3
([1, 2], [3, 4], [5, 7, 8], [6], [9], [10]) : 1
([1, 2], [3], [4, 9], [5, 6], [7, 8], [10]) : 1
([1, 9], [2], [3, 4], [5, 6], [7], [8], [10]) : 1
([1, 2], [3, 9], [4], [5, 6], [7, 8], [10]) : 1
([1, 2], [3, 5], [4, 6], [7], [8], [9, 10]) : 1
([1], [2, 7], [3, 4], [5, 6], [8], [9, 10]) : 1
([1, 2], [3, 4], [5, 6], [7], [8], [9, 10]) : 1
([1, 2], [3, 4], [5], [6], [7, 8], [9], [10]) : 1
([1, 3], [2], [4], [5, 6], [7, 8], [9], [10]) : 1
([1, 2], [3, 5], [4], [6], [7, 8], [9], [10]) : 1
"""

"""
Based on this simulation, we can see that normalizing data can
improve the performance of forward inclusion procedure.
"""
