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

# R^2 for the true group structure (#28)
truegroup_str = '([1], [2, 3], [4, 5, 6])'
print truegroup_str, ':', res_dict[truegroup_str]

'''
The top 28 group structures are listed below, where the
last group structure is the true group structure:

    1  :  ([1, 2, 3, 4, 5, 6],) 	:  0.998280984279
    2  :  ([1, 2, 3, 4, 5], [6]) 	:  0.993104524481
    3  :  ([1, 2, 3, 4, 6], [5]) 	:  0.992118473923
    4  :  ([1, 2, 3, 5, 6], [4]) 	:  0.992067571377
    5  :  ([1, 2, 3, 4], [5, 6]) 	:  0.983859855577
    6  :  ([1, 2, 3, 5], [4, 6]) 	:  0.982406579335
    7  :  ([1], [2, 3, 4, 5, 6]) 	:  0.982220382286
    8  :  ([1, 2, 3, 4], [5], [6]) 	:  0.98106128715
    9  :  ([1, 2, 3, 6], [4, 5]) 	:  0.980593601684
    10  :  ([1, 2, 3, 5], [4], [6]) 	:  0.978969940125
    11  :  ([1, 2, 3, 6], [4], [5]) 	:  0.97881449268
    12  :  ([1, 2, 3], [4, 5, 6]) 	:  0.978150484547
    13  :  ([1, 4, 5, 6], [2, 3]) 	:  0.971187526931
    14  :  ([1, 2, 3], [4, 6], [5]) 	:  0.967803824902
    15  :  ([1, 2, 3], [4], [5, 6]) 	:  0.967591407205
    16  :  ([1, 2, 3], [4, 5], [6]) 	:  0.966989376778
    17  :  ([1, 5], [2, 3, 4, 6]) 	:  0.965147585939
    18  :  ([1, 6], [2, 3, 4, 5]) 	:  0.96514654272
    19  :  ([1, 4], [2, 3, 5, 6]) 	:  0.964404716768
    20  :  ([1, 2, 3], [4], [5], [6]) 	:  0.963341433094
    21  :  ([1], [2, 3, 4, 6], [5]) 	:  0.962476638295
    22  :  ([1, 5, 6], [2, 3, 4]) 	:  0.96094095558
    23  :  ([1], [2, 3, 4, 5], [6]) 	:  0.960569735301
    24  :  ([1], [2, 3, 5, 6], [4]) 	:  0.959266766802
    25  :  ([1, 4, 6], [2, 3, 5]) 	:  0.957999996255
    26  :  ([1, 4, 5], [2, 3, 6]) 	:  0.955490081705
    27  :  ([1, 5, 6], [2, 3], [4]) 	:  0.951755741871
    28  :  ([1], [2, 3], [4, 5, 6]) 	:  0.951470439652


# Large Function Space Results in Better Fitting
# ----------------------------------------------

Splitting the last group, i.e. sigmoid(x_4 + x_5 + x_6), does
not seem to damage the goodness of fitting. In the top 11 group
structures, group structures with various ways to split the
last grouping are presented. In particular, the following ways
of splitting (x_4, x_5, x_6) gives good fitting performance (
in decreasing order):

    1. Split one variable as a univariate group, then the other
       two join the remaining variables as a single group (see
       group structure 2, 3, 4).

    2. Split two variables as a bivariate group, then the other
       one joins the remaining variables as a single group (see
       group structure 5, 6, 9).

    3. Split two variables as two univariate groups, then the
       other one joins the remaining variables as a single group
       (see group structure 8, 10, 11).

The two group structures ranked in the top 11 are group structure
1 and 7. The group structure 1 is the most flexible structure. In
the simulation study "sim_01172016", it also gave the best fitting
performance. Group structure 7 is an inherited group structure, the
good fitting performance is due to the large function space we are
using.

** Since larger function spaces usually give better performance, at
least in these two simulation studies, some penalty should be imposed
on OKGT fitting regarding the size of the function space. This is
similar to penalizing for the number of the variables in linear
regression.**

# Mis-specification Damages Fitting Performance
# ---------------------------------------------

The following two group structures correspond to the same
function spaces. However, the estimation of R^2 are drastically
different.

 ([1, 3], [2, 4], [5], [6]) : 0.6677935518
 ([1, 4], [2, 3], [5], [6]) : 0.9348431480

The group 195 is more severely mis-specified than group 197.
While group 197 has [2,3] being a correct grouping, all of the
groupings in group 195 are incorrect.

**This shows that with the same function space, the specification
of the variable partition has great impact on OKGT fitting.**
'''
