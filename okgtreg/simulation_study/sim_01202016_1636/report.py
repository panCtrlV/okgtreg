__author__ = 'panc'

import pickle
import operator
import numpy as np

lmbda = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), 10))  # evenly spaced on log-scale

'''
# Model 1
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-1.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

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
202  :  ([1], [2], [3], [4], [5], [6])  :  0.991569738028
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

## 2-based penalty
# complexityList = [ np.sum([2 ** len(g) for g in gstruct.partition])
#                    for gstruct in groupsList ]

## e-based penalty
# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## n^{n+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
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

#
# for item in zip(groupsList, r2List, complexityList, r2adjList):
#     print item
#
# for item in sorted(zip(groupsList, r2List, complexityList, r2adjList), key=operator.itemgetter(3), reverse=True):
#     print item
'''
# 2-based penalty
# ---------------
lambda	:	true_group_rank
1e-10 	:	202
8.13625304968e-10 	:	202
6.61986136885e-09 	:	202
5.38608672508e-08 	:	202
4.38225645428e-07 	:	202
3.56551474406e-06 	:	202
2.90099302101e-05 	:	202
0.000236032133143 	:	125
0.00192041716311 	:	75
0.015625 	:	75

# e-based penalty
# ---------------
lambda	:	true_group_rank
1e-10 	:	202
8.13625304968e-10 	:	202
6.61986136885e-09 	:	202
5.38608672508e-08 	:	202
4.38225645428e-07 	:	202
3.56551474406e-06 	:	201
2.90099302101e-05 	:	148
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	:	1

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	202
8.13625304968e-10 	:	202
6.61986136885e-09 	:	202
5.38608672508e-08 	:	202
4.38225645428e-07 	:	202
3.56551474406e-06 	:	200
2.90099302101e-05 	:	193
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	:	1


=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	202
8.13625304968e-10 	:	202
6.61986136885e-09 	:	202
5.38608672508e-08 	:	202
4.38225645428e-07 	:	201
3.56551474406e-06 	:	190
2.90099302101e-05 	:	73
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	:	1

'''


'''
# Model 2
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-2.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

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
203  :  ([1], [2], [3], [4], [5], [6])  :  0.347687527214
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

# complexityList = [np.sum([2 ** len(g) for g in gstruct.partition])
#                   for gstruct in groupsList]

# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
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
=== 2-based penalty ===
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
0.015625 	:	202

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
0.00192041716311 	:	201
0.015625 	:	82

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
0.00192041716311 	:	202
0.015625 	:	163

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	203
8.13625304968e-10 	:	203
6.61986136885e-09 	:	203
5.38608672508e-08 	:	203
4.38225645428e-07 	:	203
3.56551474406e-06 	:	203
2.90099302101e-05 	:	202
0.000236032133143 	:	196
0.00192041716311 	:	168
0.015625 	:	58
'''


'''
# Model 3
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-3.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

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
15  :  ([1], [2], [3], [4, 5, 6])  :  0.990084789266
'''

res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

# complexityList = [np.sum([2 ** len(g) for g in gstruct.partition])
#                   for gstruct in groupsList]

# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
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
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	15
3.56551474406e-06 	:	15
2.90099302101e-05 	:	15
0.000236032133143 	:	14
0.00192041716311 	:	4
0.015625 	:	4

# e-based penalty
# --------------
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	15
3.56551474406e-06 	:	15
2.90099302101e-05 	:	13
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	:	11

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	15
3.56551474406e-06 	:	15
2.90099302101e-05 	:	14
0.000236032133143 	:	10
0.00192041716311 	:	1
0.015625 	:	2

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	14
3.56551474406e-06 	:	11
2.90099302101e-05 	:	3
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	:	30
'''


'''
# Model 4
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-4.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

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
15  :  ([1], [2], [3], [4, 5, 6])  :  0.993972146566
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

# complexityList = [np.sum([2 ** len(g) for g in gstruct.partition])
#                   for gstruct in groupsList]

# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
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
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	15
3.56551474406e-06 	:	15
2.90099302101e-05 	:	15
0.000236032133143 	:	4
0.00192041716311 	:	14
0.015625 	:	80

# e-based penalty
# ---------------
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	15
3.56551474406e-06 	:	15
2.90099302101e-05 	:	5
0.000236032133143 	:	1
0.00192041716311 	:	59
0.015625 	:	77

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	15
3.56551474406e-06 	:	15
2.90099302101e-05 	:	15
0.000236032133143 	:	1
0.00192041716311 	:	8
0.015625 	:	62

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	12
3.56551474406e-06 	:	5
2.90099302101e-05 	:	4
0.000236032133143 	:	11
0.00192041716311 	:	77
0.015625 	:	77
'''


'''
# Model 5
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-5.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

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
45  :  ([1], [2], [3], [4, 5, 6])  :  0.986146245772
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

## 2-based penalty
# complexityList = [np.sum([2 ** len(g) for g in gstruct.partition])
#                   for gstruct in groupsList]

## e-based penalty
# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
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
lambda	:	true_group_rank
1e-10 	:	45
8.13625304968e-10 	:	45
6.61986136885e-09 	:	45
5.38608672508e-08 	:	45
4.38225645428e-07 	:	45
3.56551474406e-06 	:	44
2.90099302101e-05 	:	43
0.000236032133143 	:	22
0.00192041716311 	:	87
0.015625 	:	87

# e-based penalty
# ---------------
lambda	:	true_group_rank
1e-10 	:	45
8.13625304968e-10 	:	45
6.61986136885e-09 	:	45
5.38608672508e-08 	:	45
4.38225645428e-07 	:	44
3.56551474406e-06 	:	43
2.90099302101e-05 	:	31
0.000236032133143 	:	16
0.00192041716311 	:	77
0.015625 	:	77

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	45
8.13625304968e-10 	:	45
6.61986136885e-09 	:	45
5.38608672508e-08 	:	45
4.38225645428e-07 	:	45
3.56551474406e-06 	:	44
2.90099302101e-05 	:	41
0.000236032133143 	:	17
0.00192041716311 	:	63
0.015625 	:	63

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	45
8.13625304968e-10 	:	45
6.61986136885e-09 	:	45
5.38608672508e-08 	:	44
4.38225645428e-07 	:	43
3.56551474406e-06 	:	28
2.90099302101e-05 	:	15
0.000236032133143 	:	80
0.00192041716311 	:	77
0.015625 	:	77
'''


'''
# Model 6
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-6.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

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
15  :  ([1], [2], [3], [4, 5, 6])  :  0.999733564469
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

# complexityList = [np.sum([2 ** len(g) for g in gstruct.partition])
#                   for gstruct in groupsList]

# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
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
=== 2-based penalty ===
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	14
3.56551474406e-06 	:	9
2.90099302101e-05 	:	24
0.000236032133143 	:	82
0.00192041716311 	:	82
0.015625 	:	82

=== e-based penalty ===
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	14
4.38225645428e-07 	:	10
3.56551474406e-06 	:	10
2.90099302101e-05 	:	79
0.000236032133143 	:	79
0.00192041716311 	:	79
0.015625 	:	79

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	15
5.38608672508e-08 	:	15
4.38225645428e-07 	:	14
3.56551474406e-06 	:	9
2.90099302101e-05 	:	28
0.000236032133143 	:	64
0.00192041716311 	:	64
0.015625 	:	64

In this example, the improvement as a result of the 2-power penalty
is better than that of the e-based penalty.

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	15
8.13625304968e-10 	:	15
6.61986136885e-09 	:	14
5.38608672508e-08 	:	10
4.38225645428e-07 	:	7
3.56551474406e-06 	:	37
2.90099302101e-05 	:	79
0.000236032133143 	:	79
0.00192041716311 	:	79
0.015625 	:	79
'''


'''
# Model 7
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-7.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure (#41)
truegroup_str = '([1], [2], [3], [4, 5, 6])'
truerankid = int(np.where([k.__str__() == truegroup_str
                           for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
142  :  ([1], [2], [3], [4, 5, 6])  :  0.873747049741
'''

res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

# complexityList = [ np.sum([2 ** len(g) for g in gstruct.partition])
#                    for gstruct in groupsList ]

# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
                  for gstruct in groupsList]

# Print the tabulation of different lambda and
# the resulting rank of the true group structure
print 'lambda\t:\ttrue_group_rank'
for i in range(len(lmbda)):
    r2adjList = np.array(r2List) - lmbda[i] * np.array(complexityList)
    gstructWithR2adj_dict = dict(zip(groupsList, r2adjList))
    sorted_gstructWithR2adj = sorted(gstructWithR2adj_dict.items(),
                                     key=operator.itemgetter(1),
                                     reverse=True)
    # for i in range(len(sorted_gstructWithR2adj)):
    #     print i, ':', sorted_gstructWithR2adj[i][0], ' : ', sorted_gstructWithR2adj[i][1]
    true_id = int(np.where([item[0].__str__() == truegroup_str
                            for item in sorted_gstructWithR2adj])[0])
    print lmbda[i], '\t:\t', true_id + 1
'''
lambda	:	true_group_rank
1e-10 	:	142
8.13625304968e-10 	:	142
6.61986136885e-09 	:	142
5.38608672508e-08 	:	142
4.38225645428e-07 	:	142
3.56551474406e-06 	:	142
2.90099302101e-05 	:	142
0.000236032133143 	:	143
0.00192041716311 	:	149
0.015625 	:	151

# e-based penalty
# ---------------
lambda	:	true_group_rank
1e-10 	:	142
8.13625304968e-10 	:	142
6.61986136885e-09 	:	142
5.38608672508e-08 	:	142
4.38225645428e-07 	:	142
3.56551474406e-06 	:	142
2.90099302101e-05 	:	143
0.000236032133143 	:	145
0.00192041716311 	:	128
0.015625 	:	100

=== 2-power penalty ===
lambda	:	true_group_rank
1e-10 	:	142
8.13625304968e-10 	:	142
6.61986136885e-09 	:	142
5.38608672508e-08 	:	142
4.38225645428e-07 	:	142
3.56551474406e-06 	:	142
2.90099302101e-05 	:	142
0.000236032133143 	:	141
0.00192041716311 	:	140
0.015625 	:	96

The 2-power penalty also performs a little better than
the e-based penalty.

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	142
8.13625304968e-10 	:	142
6.61986136885e-09 	:	142
5.38608672508e-08 	:	142
4.38225645428e-07 	:	142
3.56551474406e-06 	:	141
2.90099302101e-05 	:	138
0.000236032133143 	:	128
0.00192041716311 	:	137
0.015625 	:	100
'''


'''
# Model 8
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-8.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

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
156  :  ([1], [2], [3], [4], [5], [6])  :  0.999733626801
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

# complexityList = [np.sum([2 ** len(g) for g in gstruct.partition])
#                   for gstruct in groupsList]

# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
                  for gstruct in groupsList]

# Print the tabulation of different lambda and
# the resulting rank of the true group structure
print 'lambda\t:\ttrue_group_rank'
for i in range(len(lmbda)):
    r2adjList = np.array(r2List) - lmbda[i] * np.array(complexityList)
    gstructWithR2adj_dict = dict(zip(groupsList, r2adjList))
    sorted_gstructWithR2adj = sorted(gstructWithR2adj_dict.items(),
                                     key=operator.itemgetter(1),
                                     reverse=True)
    true_id = int(np.where([item[0].__str__() == truegroup_str
                            for item in sorted_gstructWithR2adj])[0])
    print lmbda[i], '\t:\t', true_id + 1
'''
=== 2-based penalty ===
lambda	            :	true_group_rank
1e-10 	            :	156
8.13625304968e-10 	:	156
6.61986136885e-09 	:	156
5.38608672508e-08 	:	156
4.38225645428e-07 	:	155
3.56551474406e-06 	:	150
2.90099302101e-05 	:	99
0.000236032133143 	:	38
0.00192041716311 	:	38
0.015625 	        :	38

=== e-based penalty ===
lambda	            :	true_group_rank
1e-10 	            :	156
8.13625304968e-10 	:	156
6.61986136885e-09 	:	156
5.38608672508e-08 	:	156
4.38225645428e-07 	:	151
3.56551474406e-06 	:	107
2.90099302101e-05 	:	1
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	        :	1

=== 2-power penalty ===
lambda	            :	true_group_rank
1e-10 	            :	156
8.13625304968e-10 	:	156
6.61986136885e-09 	:	156
5.38608672508e-08 	:	156
4.38225645428e-07 	:	156
3.56551474406e-06 	:	135
2.90099302101e-05 	:	1
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	        :	1

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	156
8.13625304968e-10 	:	156
6.61986136885e-09 	:	153
5.38608672508e-08 	:	151
4.38225645428e-07 	:	129
3.56551474406e-06 	:	63
2.90099302101e-05 	:	1
0.000236032133143 	:	1
0.00192041716311 	:	1
0.015625 	:	1
'''


'''
# Model 9
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-9.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure (#41)
truegroup_str = '([1], [2, 3], [4, 5, 6])'
truerankid = int(np.where([k.__str__() == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
14  :  ([1], [2, 3], [4, 5, 6])  :  0.978444736553
'''
# Ranking after adding penalty
res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

# complexityList = [np.sum([2 ** len(g) for g in gstruct.partition])
#                   for gstruct in groupsList]

# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
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
lambda	:	true_group_rank
1e-10 	:	14
8.13625304968e-10 	:	14
6.61986136885e-09 	:	14
5.38608672508e-08 	:	14
4.38225645428e-07 	:	14
3.56551474406e-06 	:	14
2.90099302101e-05 	:	6
0.000236032133143 	:	8
0.00192041716311 	:	12
0.015625 	:	78

# e-based penalty
# ---------------
lambda	            :	true_group_rank
1e-10 	            :	14
8.13625304968e-10 	:	14
6.61986136885e-09 	:	14
5.38608672508e-08 	:	14
4.38225645428e-07 	:	14
3.56551474406e-06 	:	7
2.90099302101e-05 	:	4
0.000236032133143 	:	14
0.00192041716311 	:	82
0.015625 	        :	98

=== 2-power penalty ===
lambda	            :	true_group_rank
1e-10 	            :	14
8.13625304968e-10 	:	14
6.61986136885e-09 	:	14
5.38608672508e-08 	:	14
4.38225645428e-07 	:	14
3.56551474406e-06 	:	14
2.90099302101e-05 	:	8
0.000236032133143 	:	14
0.00192041716311 	:	70
0.015625 	        :	98

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	14
8.13625304968e-10 	:	14
6.61986136885e-09 	:	14
5.38608672508e-08 	:	13
4.38225645428e-07 	:	8
3.56551474406e-06 	:	3
2.90099302101e-05 	:	12
0.000236032133143 	:	14
0.00192041716311 	:	82
0.015625 	:	98
'''


'''
# Model 10
# =======
'''
resfile = open("okgtreg/simulation/sim_01202016_1636/script-model-10.pkl", 'rb')
res_dict = pickle.load(resfile)
resfile.close()

# Sort group structures by R^2's in decreasing order
sortedRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
for (k, v) in sortedRes:
    counter += 1
    print counter, ' : ', k.__str__(), ' : ', v

# R^2 for the true group structure (#41)
truegroup_str = '([1], [2, 3], [4, 5, 6])'
truerankid = int(np.where([k.__str__() == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
2  :  ([1], [2, 3], [4, 5, 6])  :  0.997711038756
'''

res_list = res_dict.items()
groupsList = [k for (k, v) in res_list]
r2List = [v for (k, v) in res_list]

# complexityList = [np.sum([2 ** len(g) for g in gstruct.partition])
#                   for gstruct in groupsList]

# complexityList = [np.sum([np.exp(len(g)) for g in gstruct.partition])
#                   for gstruct in groupsList]

## 2-power penalty
# print("=== 2-power penalty ===")
# complexityList = [np.sum([len(g) ** 2 for g in gstruct.partition])
#                   for gstruct in groupsList]

## d^{d+2} penalty
print("=== d^{d+2} penalty ===")
complexityList = [np.sum([len(g) ** len(g) for g in gstruct.partition])
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
0.015625 	:	77

# e-based penalty
# ---------------
lambda	:	true_group_rank
1e-10 	:	2
8.13625304968e-10 	:	2
6.61986136885e-09 	:	2
5.38608672508e-08 	:	2
4.38225645428e-07 	:	2
3.56551474406e-06 	:	2
2.90099302101e-05 	:	1
0.000236032133143 	:	2
0.00192041716311 	:	78
0.015625 	:	97

=== 2-power penalty ===
lambda	            :	true_group_rank
1e-10 	            :	2
8.13625304968e-10 	:	2
6.61986136885e-09 	:	2
5.38608672508e-08 	:	2
4.38225645428e-07 	:	2
3.56551474406e-06 	:	2
2.90099302101e-05 	:	1
0.000236032133143 	:	2
0.00192041716311 	:	20
0.015625 	        :	97

=== d^{d+2} penalty ===
lambda	:	true_group_rank
1e-10 	:	2
8.13625304968e-10 	:	2
6.61986136885e-09 	:	2
5.38608672508e-08 	:	2
4.38225645428e-07 	:	2
3.56551474406e-06 	:	1
2.90099302101e-05 	:	1
0.000236032133143 	:	2
0.00192041716311 	:	78
0.015625 	:	97
'''
