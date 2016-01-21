__author__ = 'panc'

import pickle
import operator
import numpy as np

from okgtreg import *


# The results contains all group structures and
# the corresponding R^2's. They are saved in a
# dictionary.
resfile = open("okgtreg/simulation/sim_01172016/script.py.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure
allgroups = res_dict.keys()
true_id = np.argmax([group.__str__() == Group([1], [2, 3], [4, 5, 6]).__str__() for group in allgroups])  # 95
print true_id
truegroup = allgroups[true_id]

print truegroup, ' : ', res_dict[truegroup]

'''
The true group structure is the No.4 in the sorted results.

The four group structures that have higher R^2 are:

 Group structure ([1, 2, 3, 4, 5, 6],), 0.9543248157662435
 Group structure ([1, 2, 3], [4, 5, 6]), 0.9515600003731547
 Group structure ([1], [2, 3, 4, 5, 6]), 0.9510543999814678
 Group structure ([1, 4, 5, 6], [2, 3]), 0.9501499675379735

It can be noticed that the above four group structures are
inherited group structures by using the terminology in our
OKGT paper. In this simulation example, the true group structure
is:

    [1], [2,3], [4,5,6].

So there are only four inherited group structures which are
listed above. In this example, they all perform better than
the true group structure.

The result of this simulation seems promising. The difference
of R^2 between the top six group structures are:

    -2.76481539e-03,  -5.05600392e-04,  -9.04432443e-04,
    -3.21405809e-04,  -1.62512254e-02

where the last value is the difference of R^2 between the true
group structure and the top ranked incorrect group structure
([1, 2, 3, 4, 5], [6]). The magnitude of the difference is one
 order higher than the largest difference between the top four
 group structures. It seems the different is significant.

In order to test the significance of the differences, bootstrap
is used.

'''

# Complexity vs R2
import matplotlib.pyplot as plt

sortedR2 = [v for (k, v) in sortedRes]
sortedComplexity = [np.sum([2 ** len(g) for g in k.partition]) for (k, v) in sortedRes]

plt.title(r"Complexity of group structure against $R^2$" + '\n' +
          r"Red circle is the true group structure")
plt.scatter(sortedR2, sortedComplexity)
plt.scatter(sortedR2[4], sortedComplexity[4], s=300, facecolors='none', edgecolors='r')
plt.xlabel(r"$R^2$")
plt.ylabel("Complexity")

'''
Bootstrap gives the estimates of standard deviation of the
difference between the estimated $R^2$'s.
'''
# First order difference of ordered R^2
np.diff(sorted(res_dict.values(), reverse=True))

for i in range(100):
    filename = 'bootstrap.py-' + str(i + 1) + '.pkl'
    file = open('okgtreg/simulation/sim_01172016/bootstrap/' + filename, 'rb')
    bootstrap_res = pickle.load(file)
    file.close()

    pass
