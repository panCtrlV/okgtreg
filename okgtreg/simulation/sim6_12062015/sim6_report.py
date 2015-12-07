import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Read pickled results into memory
pkl_file = open("okgtreg/simulation/sim6_12062015/estimatedR2s.pkl", 'rb')
r2s = pickle.load(pkl_file)
pkl_file.close()

r2s_series = pd.Series(r2s)
r2s_series.describe()
"""
count    100.000000
mean       0.889915
std        0.000481
min        0.888392
25%        0.889704
50%        0.889924
75%        0.890230
max        0.890985
"""

fig, axes = plt.subplots(1, 2)
fig.suptitle(r"Distribution of $\hat{R^2}$")
axes[0].hist(r2s, 15)
axes[1].boxplot(r2s)

# Summarize group structures
pkl_file = pkl_file = open("okgtreg/simulation/sim6_12062015/estimatedGroupStructures.pkl", 'rb')
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
Counter({(1,): 71,
         (1, 2): 4,
         (1, 2, 4): 1,
         (1, 4): 24,
         (2, 3): 50,
         (2, 3, 5): 1,
         (2, 4): 44,
         (3,): 15,
         (3, 5): 34,
         (4,): 20,
         (4, 5): 11,
         (5,): 54,
         (6,): 100,
         (7,): 100,
         (8,): 100})

Though the true group structure has 6, 7, 8 as a single group, none of the simulation
recovers this structure. Some group structures are detected for 1, 2, 3, 4, 5.

Recall, we use the same set of X (seed is fixed) as those in sim5. The only difference
from sim5 is that the true group, [6,7,8], is multiplied by 100 when they are used to
generate the response. **This means that the magnitude of each transformation will affect
the group structure detection.**
"""
