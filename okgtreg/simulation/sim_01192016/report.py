__author__ = 'panc'

from okgtreg import *

import pickle

# The results contains all group structures and
# the corresponding R^2's. They are saved in a
# dictionary. The keys of the dictionary are the
# string representation of the group structures.as
resfile = open("okgtreg/simulation/sim_01192016/script.py.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

# Sort group structures by R^2's in decreasing order
import operator

sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
for i in range(len(sortedRes)):
    print i + 1, ' : ', sortedRes[i][0], '\t: ', sortedRes[i][1]

# R^2 for the true group structure (#41)
truegroup_str = '([1], [2, 3], [4, 5, 6])'
truerankid = int(np.where([k == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]

'''
Among the ranked group structures in terms of the estimated
$R^2$'s. The true group structure is the 41-st on the list.

Overall, OKGT with gaussian kernel does not provides a good
fitting. The largest $R^2$ is 0.8.

By inspecting the top 40 group structures, the grouping of
covariates 4, 5, 6 plays an active role in forming different
group structures. In the top 15 group structures, the covriates
1-3 are always grouped together, while the partitions of 4-6
causes different group structures.

Another observation is that the binding of the covariates 2
and 3 is strong. They are grouped together in the top 40 group
structures. Actually, they are together until the 52-nd group
structure.
'''

# Calculate the proposed complexity
import re
import numpy as np
import matplotlib.pyplot as plt


def calculateComplexityFromKey(key):
    groupstrlist = re.findall('\[[^\]]*\]', key)
    grouptuple = tuple([[int(d) for d in re.findall('\d', s)] for s in groupstrlist])
    group = Group(*grouptuple)
    groupcomplexity = np.sum([2 ** len(g) for g in group.partition])
    return groupcomplexity


sortedComplexity = [calculateComplexityFromKey(k) for (k, v) in sortedRes]

## plot the ranked complexities
plt.title("Complexities of the ranked group structures")
plt.plot(sortedComplexity)  # order of complexity in terms of fitting

# Plot complexity vs R2
sortedR2 = [v for (k, v) in sortedRes]

plt.title(r"Complexity of group structure against $R^2$" + '\n' +
          r"Red circle is the true group structure")
plt.scatter(sortedR2, sortedComplexity)
plt.scatter(sortedR2[truerankid], sortedComplexity[truerankid],
            s=300, facecolors='none', edgecolors='r')
plt.xlabel(r"$R^2$")
plt.ylabel("Complexity")

'''
We plot the complexities for different group structure against
the estimated $R^2$. It can be seen that this is an upward
trend relationship between the complexities and the estimated
$R^2$. However, the relationship is not monotone. The estimated
$R^2$'s for the group structures with the same complexity cover
multiple ranges from low to high. In the figure, a group structure
with complexity around 12 can beat that with complexity around 35.
'''
