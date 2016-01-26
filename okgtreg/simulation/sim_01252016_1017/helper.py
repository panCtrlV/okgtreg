__author__ = 'panc'

import numpy as np
import collections

from okgtreg.Group import Group


def mapGroupToDictionary(g):
    """
    Map a Group object to a dictionary.

    :type g: Group
    :param g:
    :return:
    """
    g_dict = {}
    for i in np.arange(g.size) + 1:
        g_dict[tuple(g[i])] = 1
    return g_dict


def printGroupingFrequency(groupList, sort=True):
    groupDictList = [mapGroupToDictionary(group) for group in groupList]
    counter = collections.Counter([x for g in groupDictList for x in g])
    if sort:
        counter = counter.most_common()
        for item in counter:
            print item[0], ':', item[1]
    else:
        for item in counter.items():
            print item[0], ':', item[1]
    return counter


def printGroupFrequency(groupList, sort=True):
    groupDictList = [mapGroupToDictionary(group) for group in groupList]
    counter = collections.Counter(tuple(g.keys()) for g in groupDictList)
    if sort:
        counter = counter.most_common()
        for item in counter:
            print Group(*tuple(list(i) for i in item[0])), ':', item[1]
    else:
        for item in counter.items():
            print Group(*tuple(list(i) for i in item[0])), ':', item[1]
    return counter
