import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Read pickled results into memory
pkl_file = open("okgtreg/simulation/sim7_12062015/estimatedR2s.pkl", 'rb')
r2s = pickle.load(pkl_file)
pkl_file.close()

r2s_series = pd.Series(r2s)
r2s_series.describe()
"""
count    100.000000
mean       0.987390
std        0.001139
min        0.984369
25%        0.986584
50%        0.987339
75%        0.988173
max        0.989952
"""

fig, axes = plt.subplots(1, 2)
fig.suptitle(r"Distribution of $\hat{R^2}$")
axes[0].hist(r2s, 15)
axes[1].boxplot(r2s)

# Summarize group structures
pkl_file = pkl_file = open("okgtreg/simulation/sim7_12062015/estimatedGroupStructures.pkl", 'rb')
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
         (5,): 97,
         (5, 6): 1,
         (5, 8): 2,
         (6,): 54,
         (6, 7): 25,
         (6, 8): 20,
         (7,): 59,
         (7, 8): 16,
         (8,): 62})
"""