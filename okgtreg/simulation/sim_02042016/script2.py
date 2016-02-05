__author__ = 'panc'

import numpy as np
import pickle
import operator

from okgtreg import *
from okgtreg.simulation.sim_02042016.helper import rkhsCapacity


# ================================================
# Helper functions to process simulation results
# ================================================
def sortGroupStructures(res_dict, decreasing=True, printing=True):
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


def sortGroupStructuresWithPenalty(res_dict, mu, a, printing=False):
    sortedRes = sortGroupStructures(res_dict, printing=printing)
    gList = [k for (k, v) in sortedRes]
    r2 = np.array([v for (k, v) in sortedRes])
    penalty = np.array([rkhsCapacity(k, a) for (k, v) in sortedRes]) * mu
    pR2 = r2 - penalty
    pResDict = dict(zip(gList, pR2))
    pSortedRes = sortGroupStructures(pResDict, printing=printing)
    return pSortedRes


def rankWithPenalty(res_dict, group, mu, a):
    sortedRes = sortGroupStructures(res_dict, printing=False)
    gList = [k for (k, v) in sortedRes]
    r2 = np.array([v for (k, v) in sortedRes])
    penalty = np.array([rkhsCapacity(k, a) for (k, v) in sortedRes]) * mu
    pR2 = r2 - penalty
    pResDict = dict(zip(gList, pR2))
    return rank(group, pResDict)


'''
Rank group structure after imposing penalty
for each pair of (mu, a)
'''

# Tuning parameter \mu
# mu = 1e-4
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), 10))
aList = np.arange(1, 11)

# Unpickle result file
filename = "script-model-4.pkl"
with open("okgtreg/simulation/sim_02042016/" + filename, 'rb') as f:
    resDict = pickle.load(f)

# Sort the un-penalized results
sortedRes = sortGroupStructures(resDict)
# Rank of the true group structure
tgroup = Group([1], [2], [3], [4], [5], [6])
print rank(tgroup, resDict)

# Rank of the true group structures for all
# (mu, a) pairs
for mu in muList:
    for a in aList:
        pSortedRes = sortGroupStructuresWithPenalty(resDict, mu, a)
        pRank = rankWithPenalty(resDict, tgroup, mu, a)
        print("mu = %.11f, a = %.02f : pR2 = %.10f, rank = %d" %
              (mu, a, pSortedRes[pRank - 1][1], pRank))
