__author__ = 'panc'

import pickle
import operator
import numpy as np

lmbda = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), 10))  # evenly spaced on log-scale

'''
# Model 2
# =======
'''
res_file = open("okgtreg/simulation/sim_01212016_2043/script-model-2.pkl", 'rb')
res_dict = pickle.load(res_file)
res_file.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure (#41)
truegroup_str = '([1], [2], [3], [4], [5], [6])'
truerankid = int(np.where([k.__str__() == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
203  :  ([1], [2], [3], [4], [5], [6])  :  0.906026768147
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

## e-based penalty
print("=== e-based penalty ===")
complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
                  for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

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
1e-10 	:	203
8.13625304968e-10 	:	203
6.61986136885e-09 	:	203
5.38608672508e-08 	:	203
4.38225645428e-07 	:	203
3.56551474406e-06 	:	203
2.90099302101e-05 	:	203
0.000236032133143 	:	203
0.00192041716311 	:	177
0.015625 	:	21

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	203
8.13625304968e-10 	:	203
6.61986136885e-09 	:	203
5.38608672508e-08 	:	203
4.38225645428e-07 	:	203
3.56551474406e-06 	:	203
2.90099302101e-05 	:	203
0.000236032133143 	:	203
0.00192041716311 	:	203
0.015625 	:	18
'''

'''
# Model 5
# =======

With cubic transformation for Y.
'''
res_file = open("okgtreg/simulation/sim_01212016_2043/script-model-5.pkl", 'rb')
res_dict = pickle.load(res_file)
res_file.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure (#41)
truegroup_str = '([1], [2], [3], [4, 5, 6])'
truerankid = int(np.where([k.__str__() == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
111  :  ([1], [2], [3], [4, 5, 6])  :  0.999042981727
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
1e-10 	:	111
8.13625304968e-10 	:	111
6.61986136885e-09 	:	111
5.38608672508e-08 	:	111
4.38225645428e-07 	:	111
3.56551474406e-06 	:	110
2.90099302101e-05 	:	74
0.000236032133143 	:	79
0.00192041716311 	:	80
0.015625 	:	80

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	111
8.13625304968e-10 	:	111
6.61986136885e-09 	:	111
5.38608672508e-08 	:	111
4.38225645428e-07 	:	111
3.56551474406e-06 	:	111
2.90099302101e-05 	:	109
0.000236032133143 	:	5
0.00192041716311 	:	65
0.015625 	:	65

While fitting OKGT with Laplace kernel in this example,
using 2-power penalty performs significantly better than
the e-based penalty.

The combination of laplace and 2-power penalty performs
better than the simulation study on the same model (model 5)
in "sim_01202016_1636".
'''

'''
# Model 6
# =======
'''
res_file = open("okgtreg/simulation/sim_01212016_2043/script-model-6.pkl", 'rb')
res_dict = pickle.load(res_file)
res_file.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure (#41)
truegroup_str = '([1], [2], [3], [4, 5, 6])'
truerankid = int(np.where([k.__str__() == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
8  :  ([1], [2], [3], [4, 5, 6])  :  0.999885315435
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

## e-based penalty
print("=== e-based penalty ===")
complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
                  for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

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
1e-10 	:	8
8.13625304968e-10 	:	8
6.61986136885e-09 	:	8
5.38608672508e-08 	:	8
4.38225645428e-07 	:	7  *
3.56551474406e-06 	:	23
2.90099302101e-05 	:	77
0.000236032133143 	:	77
0.00192041716311 	:	77
0.015625 	:	77

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	8
8.13625304968e-10 	:	8
6.61986136885e-09 	:	8
5.38608672508e-08 	:	8
4.38225645428e-07 	:	8
3.56551474406e-06 	:	6  *
2.90099302101e-05 	:	43
0.000236032133143 	:	64
0.00192041716311 	:	64
0.015625 	:	64


By changing the generating distribution of $X$ from
standard t to standard normal, the ranking of the true
group structure becomes slightly better than the same
example in "sim_01202016_1636". But the improvement is
not significant.

Two penalties are used. The 2-power penalty results in a
better ranking than the e-based penalty. However, the
difference is not significant.
'''

'''
# Model 7
# =======
'''
res_file = open("okgtreg/simulation/sim_01212016_2043/script-model-7.pkl", 'rb')
res_dict = pickle.load(res_file)
res_file.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure (#41)
truegroup_str = '([1], [2], [3], [4, 5, 6])'
truerankid = int(np.where([k.__str__() == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
111  :  ([1], [2], [3], [4, 5, 6])  :  0.998817201802
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
1e-10 	:	111
8.13625304968e-10 	:	111
6.61986136885e-09 	:	111
5.38608672508e-08 	:	111
4.38225645428e-07 	:	111
3.56551474406e-06 	:	110
2.90099302101e-05 	:	76
0.000236032133143 	:	80
0.00192041716311 	:	81
0.015625 	:	81

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	111
8.13625304968e-10 	:	111
6.61986136885e-09 	:	111
5.38608672508e-08 	:	111
4.38225645428e-07 	:	111
3.56551474406e-06 	:	111
2.90099302101e-05 	:	106
0.000236032133143 	:	8
0.00192041716311 	:	66
0.015625 	:	66


In this example, changing kernel to laplace can improves the
rank of the true model. But the improvement is not dramatic.
Even after using the e-based penalty, the ranking is not improved
by much.

However, when we use the combination of laplace kernel and
2-power penalty, the true group structure is pushed to the
8-th place, which shows dramatic improvement.
'''
