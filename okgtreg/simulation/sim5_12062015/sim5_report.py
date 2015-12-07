import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Read pickled results into memory
pkl_file = open("okgtreg/simulation/sim5_12062015/estimatedR2s.pkl", 'rb')
r2s = pickle.load(pkl_file)
pkl_file.close()

r2s_series = pd.Series(r2s)
r2s_series.describe()
"""
count    100.000000
mean       0.986146
std        0.000478
min        0.985211
25%        0.985801
50%        0.986114
75%        0.986453
max        0.987352
"""

fig, axes = plt.subplots(1, 2)
fig.suptitle(r"Distribution of $\hat{R^2}$")
axes[0].hist(r2s, 15)
axes[1].boxplot(r2s)


# TODO: Use a social network to visualize the group structures of the covariates?

"""
NetworkX provides basic functionality for visualizing graphs, but its main goal
is to enable graph analysis rather than perform graph visualization. In the future,
graph visualization functionality may be removed from NetworkX or only available
as an add-on package.
"""

pkl_file = pkl_file = open("okgtreg/simulation/sim5_12062015/estimatedGroupStructures.pkl", 'rb')
groups = pickle.load(pkl_file)
pkl_file.close()

# ----------------------
# Count group frequency
# ----------------------
# That is, for each group structure, construct a dictionary where the
# set of covariates in a group is the key, and '1' is the value. Then
# combine all dictionaries through group by key and the counts are added.
def mapGroupToDictionary(g):
    """
    Map a Group object to a dictionary.

    :type g: Group
    :param g:
    :return:
    """
    g_dict = {}
    for i in np.arange(g.size) + 1:
        g_dict[tuple(g.getPartition(i))] = 1
    return g_dict

groupDictList = [mapGroupToDictionary(g) for g in groups]

import collections
counter=collections.Counter([x for g in groupDictList for x in g])
counter
"""
Counter({(1,): 100,
         (2,): 100,
         (3,): 100,
         (4,): 100,
         (5,): 100,
         (6,): 66,
         (6, 7): 19,
         (6, 8): 15,
         (7,): 15,
         (7, 8): 66,
         (8,): 19})

Though the true model has 6, 7, 8 as a single group, the simulation does NOT
recover this group structure. The simulation shows that the procedure could
detect the pair-wise grouping, i.e. [6,7], [7,8], and [6,8], most of time.
"""

# uniqueKeys = np.unique([x for g in groupDictList for x in g])
# sum(item[(6,)] for item in groupDictList)
#
# [sum(item[key] for item in groupDictList) for key in uniqueKeys]