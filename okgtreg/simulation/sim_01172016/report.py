__author__ = 'panc'

from okgtreg import *

import pickle

# The results contains all group structures and
# the corresponding R^2's. They are saved in a
# dictionary.
resfile = open("okgtreg/simulation/sim_01172016/script.py.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

# Sort group structures by R^2's in decreasing order
import operator

sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)

# True group
import numpy as np

allgroups = res_dict.keys()
true_id = np.argmax([group.__str__() == Group([1], [2, 3], [4, 5, 6]).__str__() for group in allgroups])  # 95
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
'''
