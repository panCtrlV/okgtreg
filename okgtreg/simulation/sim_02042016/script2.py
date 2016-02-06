__author__ = 'panc'

import numpy as np
import pickle
import operator

from okgtreg.Group import Group
from okgtreg.simulation.sim_02042016.helper import rkhsCapacity, ptopCapacity, etopCapacity, pto2Capacity

# ================================================
# Helper functions to process simulation results
# ================================================
def sortGroupStructures(res_dict, decreasing=True, printing=False):
    """
    Sort the result dictionary unpicked from the simulation
    result files, according to the values of R^2.

    :type res_dict: dict
    :param res_dict: simulation results for exhaustive okgt fitting
                     for a model with small number of covariates. The
                     key is a Group structure object, and the value
                     is the estimated R^2.

    :rtype: bool
    :return:
    """
    sortRes = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=decreasing)
    if printing:
        counter = 0
        for (k, v) in sortRes:
            counter += 1
            print counter, ' : ', k.__str__(), ' : ', v
    return sortRes


def rank(gstruct, res_dict, decreasing=True):
    """
    Given a dictionary of exhaustive OKGT fitting for a model.
    Return the rank (in a decreasingly sorted list) of a given
    group structure.

    :type gstruct: Group
    :param gstruct: a group structure specified by the user

    :type res_dict: dict
    :param res_dict: simulation results for exhaustive okgt fitting
                     for a model with small number of covariates. The
                     key is a Group structure object, and the value
                     is the estimated R^2.

    :type decreasing: bool
    :param decreasing: if the group structures in the result dictionary
                       is sorted in decreasing order according to R^2
    :return:
    """
    # gstruct_str = gstruct.__str__()
    sortedGroupStructures = sortGroupStructures(res_dict, printing=False)
    pos = int(np.where([k == gstruct for (k, v) in sortedGroupStructures])[0])
    return pos + 1


def sortGroupStructuresWithPenalty(res_dict, capacity_id, mu, a=None, printing=False):
    sortedRes = sortGroupStructures(res_dict, printing=False)
    gList = [k for (k, v) in sortedRes]
    r2 = np.array([v for (k, v) in sortedRes])
    if capacity_id == 1:
        penalty = np.array([rkhsCapacity(k, a) for (k, v) in sortedRes]) * mu
    elif capacity_id == 2:
        penalty = np.array([ptopCapacity(k) for (k, v) in sortedRes]) * mu
    elif capacity_id == 3:
        penalty = np.array([etopCapacity(k) for (k, v) in sortedRes]) * mu
    elif capacity_id == 4:
        penalty = np.array([pto2Capacity(k) for (k, v) in sortedRes]) * mu
    else:
        raise NotImplementedError("** Capacity function not inplemented ! **")
    pR2 = r2 - penalty
    pResDict = dict(zip(gList, pR2))
    pSortedRes = sortGroupStructures(pResDict, printing=printing)
    return pSortedRes


# for mu in muList:
#     print sortGroupStructuresWithPenalty(resDict, 2)


def rankWithPenalty(res_dict, group, capacity_id, mu, a=None):
    sortedRes = sortGroupStructures(res_dict, printing=False)
    gList = [k for (k, v) in sortedRes]
    r2 = np.array([v for (k, v) in sortedRes])
    if capacity_id == 1:
        penalty = np.array([rkhsCapacity(k, a) for (k, v) in sortedRes]) * mu
    elif capacity_id == 2:
        penalty = np.array([ptopCapacity(k) for (k, v) in sortedRes]) * mu
    elif capacity_id == 3:
        penalty = np.array([etopCapacity(k) for (k, v) in sortedRes]) * mu
    elif capacity_id == 4:
        penalty = np.array([pto2Capacity(k) for (k, v) in sortedRes]) * mu
    else:
        raise NotImplementedError("** Capacity function not inplemented ! **")
    pR2 = r2 - penalty
    pResDict = dict(zip(gList, pR2))
    return rank(group, pResDict)


# True group structures for model 1 - 4
tgroups = {1: Group([1], [2], [3], [4], [5], [6]),
           2: Group([1, 2, 3], [4, 5, 6]),
           3: Group([1, 2], [3, 4], [5, 6]),
           4: Group([1, 2, 3, 4, 5, 6])}


# Tuning parameter \mu
# mu = 1e-4
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), 10))
aList = np.arange(1, 11)

'''
Rank group structure after imposing penalty
for each pair of (mu, a)
'''
# Unpickle result file
model_id = 3
filename = "script-model-" + str(model_id) + ".pkl"
with open("okgtreg/simulation/sim_02042016/" + filename, 'rb') as f:
    resDict = pickle.load(f)

tgroup = tgroups[model_id]

# Sort the un-penalized results
sortedRes = sortGroupStructures(resDict)
print rank(tgroup, resDict), ":", tgroup

# Rank of the true group structures for all
# (mu, a) pairs
for mu in muList:
    for a in aList:
        pSortedRes = sortGroupStructuresWithPenalty(resDict, 1, mu, a)
        pRank = rankWithPenalty(resDict, tgroup, 1, mu, a)
        print("mu = %.11f, a = %.02f : pR2 = %.10f, rank = %d" %
              (mu, a, pSortedRes[pRank - 1][1], pRank))
        # print("mu = %.11f, a = %.02f : pR2 = %s, rank = %d" %
        #       (mu, a, pSortedRes[pRank - 1][0], pRank))

# Use p2p capacity measure
# Rank of the true group structures for all
# mu values
for mu in muList:
    pSortedRes = sortGroupStructuresWithPenalty(resDict, 2, mu)
    pRank = rankWithPenalty(resDict, tgroup, 2, mu)
    print("mu = %.11f, a = None : pR2 = %.10f, rank = %d" %
          (mu, pSortedRes[pRank - 1][1], pRank))

# Use "etop" capacity measure
# Rank of the true group structures for all
# mu values
for mu in muList:
    pSortedRes = sortGroupStructuresWithPenalty(resDict, 3, mu)
    pRank = rankWithPenalty(resDict, tgroup, 3, mu)
    print("mu = %.11f, a = None : pR2 = %.10f, rank = %d" %
          (mu, pSortedRes[pRank - 1][1], pRank))

# Use "pto2" capacity measure
# Rank of the true group structures for all
# mu values
for mu in muList:
    pSortedRes = sortGroupStructuresWithPenalty(resDict, 4, mu)
    pRank = rankWithPenalty(resDict, tgroup, 4, mu)
    print("mu = %.11f, a = None : pR2 = %.10f, rank = %d" %
          (mu, pSortedRes[pRank - 1][1], pRank))
