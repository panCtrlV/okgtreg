__author__ = 'panc'

import pickle
import operator
import numpy as np

'''
Model 1
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

'''
Model 2
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

'''
Model 3
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

'''
# Model 4
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
truerankid = int(np.where([k.__str__() == truegroup_str for (k, v) in sortedRes])[0])
print sortedRes[truerankid]
'''
142  :  ([1], [2], [3], [4, 5, 6])  :  0.873747049741
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
