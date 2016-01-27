__author__ = 'panc'

import numpy as np
import collections

from okgtreg.Group import Group


def mapGroupToDictionary(g):
    """
    Map a Group object to a dictionary.
    e.g. ([1,2,3], [4,5,6]) => {(1,2,3):1, (4,5,6):1}

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
    # groupDictList = [mapGroupToDictionary(group) for group in groupList]
    groupStrList = [group.__str__() for group in groupList]
    counter = collections.Counter(groupStrList)
    if sort:
        counter = counter.most_common()
        for item in counter:
            print item[0], ':', item[1]
    else:
        for item in counter.items():
            print item[0], ':', item[1]
    return counter
