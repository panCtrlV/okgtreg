__author__ = 'panc'

import numpy as np
import collections
import pickle

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


def groupingFrequency(groupList, sort=True):
    groupDictList = [mapGroupToDictionary(group) for group in groupList]
    counter = collections.Counter([x for g in groupDictList for x in g])
    counter = counter.most_common() if sort else counter
    # if sort:
    #     counter = counter.most_common()
    # for item in counter:
    #     print item[0], ':', item[1]
    # else:
    #     for item in counter.items():
    #         print item[0], ':', item[1]
    return counter


def groupFrequency(groupList, sort=True):
    groupStrList = [group.__str__() for group in groupList]
    counter = collections.Counter(groupStrList)
    counter = counter.most_common() if sort else counter
    # if sort:
    #     counter = counter.most_common()
    #     for item in counter:
    #         print Group(*tuple(list(i) for i in item[0])), ':', item[1]
    # else:
    #     for item in counter.items():
    #         print Group(*tuple(list(i) for i in item[0])), ':', item[1]
    return counter


def reportResults(dirname, filename, n=10):
    with open(dirname + filename, 'rb') as f:
        res = pickle.load(f)

    groupList = res.keys()

    groupingCounter = groupingFrequency(groupList)
    if n is None:
        reportSize1 = len(groupingCounter)
    else:
        reportSize1 = len(groupingCounter) if n > len(groupingCounter) else n

    print("=== Top %d out of %d most frequent groupings ===" %
          (reportSize1, len(groupingCounter)))
    for i in range(reportSize1):
        item = groupingCounter[i]
        print("%d : %s : %d" % (i + 1, item[0], item[1]))

    print(" ")

    groupCounter = groupFrequency(groupList)
    if n is None:
        reportSize2 = len(groupCounter)
    else:
        reportSize2 = len(groupCounter) if n > len(groupCounter) else n

    print("=== Top %d out of %d most frequent group structures ===" %
          (reportSize2, len(groupCounter)))
    for i in range(reportSize2):
        item = groupCounter[i]
        print("%d : %s : %d" % (i + 1, item[0], item[1]))

    return True
