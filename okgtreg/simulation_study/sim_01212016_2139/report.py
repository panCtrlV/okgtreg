__author__ = 'panc'

import pickle
import operator
import numpy as np

# Grid of values for the tuning parameter
lmbda = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), 10))  # evenly spaced on log-scale

# Un-pickle results
res_file = open("okgtreg/simulation/sim_01212016_2139/script.py.pkl", 'rb')
res_dict = pickle.load(res_file)
res_file.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure (#41)
truegroup_str = '([1, 2, 3], [4, 5, 6])'
truerankid = int(np.where([k.__str__() == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
2  :  ([1, 2, 3], [4, 5, 6])  :  0.825326988289
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

## e-based penalty
# print("=== e-based penalty ===")
# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
print("=== 2-power penalty ===")
complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
                  for gstruct in groupsList]

# Print the tabulation of different lambda and
# the resulting rank of the true group structure
print 'lambda\t:\ttrue_group_rank'
for i in range(len(lmbda)):
    r2adjList = np.array(r2List) - lmbda[i] * np.array(complexityList)
    gstructWithR2adj_dict = dict(zip(groupsList, r2adjList))
    sorted_gstructWithR2adj = sorted(gstructWithR2adj_dict.items(), key=operator.itemgetter(1), reverse=True)
    # for i in range(len(sorted_gstructWithR2adj)):
    #     print i, ':', sorted_gstructWithR2adj[i][0], ' : ', sorted_gstructWithR2adj[i][1]
    true_id = int(np.where([item[0].__str__() == truegroup_str
                            for item in sorted_gstructWithR2adj])[0])
    print lmbda[i], '\t:\t', true_id + 1
'''
=== e-based penalty ===
lambda	:	true_group_rank
1e-10 	:	2
8.13625304968e-10 	:	2
6.61986136885e-09 	:	2
5.38608672508e-08 	:	2
4.38225645428e-07 	:	2
3.56551474406e-06 	:	2
2.90099302101e-05 	:	1
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	:	1

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	2
8.13625304968e-10 	:	2
6.61986136885e-09 	:	2
5.38608672508e-08 	:	2
4.38225645428e-07 	:	2
3.56551474406e-06 	:	2
2.90099302101e-05 	:	2
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	:	1


Using a model with quadratic transformations on covaraites
and fitting with a degree-2 polynomial kernel results in a
good ranking of the true group structure. By imposing penalties,
the true group structure is further improved to be the top ranked.
'''
