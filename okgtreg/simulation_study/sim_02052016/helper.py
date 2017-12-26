__author__ = 'panc'

import numpy as np
import collections
import pickle
import datetime

from okgtreg.Group import Group
from okgtreg.Parameters import Parameters
from okgtreg.Data import ParameterizedData


# Generates set partitions recursively
# Copied from: https://compprog.wordpress.com/2007/10/15/generating-the-partitions-of-a-set/#comment-465
def partitions(set_):
    if not set_:
        yield []
        return
    for i in xrange(2 ** len(set_) / 2):
        parts = [set(), set()]
        for item in set_:
            parts[i & 1].add(item)
            i >>= 1
        for b in partitions(parts[1]):
            yield [parts[0]] + b


# Capacity functions
# Direct sum RKHS capacity for a given group structure
def rkhsCapacity(group, a):
    return sum([a ** len(g) for g in group.partition])


def ptopCapacity(group):
    return sum([len(g) ** len(g) for g in group.partition])


def etopCapacity(group):
    return sum([np.e ** len(g) for g in group.partition])


def pto2Capacity(group):
    return sum([len(g) ** 2 for g in group.partition])


def selectCapacity(id):
    capacities = {1: rkhsCapacity,
                  2: ptopCapacity,
                  3: etopCapacity,
                  4: pto2Capacity}
    return capacities[id]


def _mapGroupToDictionary(g):
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
    groupDictList = [_mapGroupToDictionary(group) for group in groupList]
    counter = collections.Counter([x for g in groupDictList for x in g])
    counter = counter.most_common() if sort else counter
    return counter


def groupFrequency(groupList, sort=True):
    groupStrList = [group.__str__() for group in groupList]
    counter = collections.Counter(groupStrList)
    counter = counter.most_common() if sort else counter
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


# Print the current timestamp,
#   e.g. ['2016-02-26', '09:11:25.203191']
#   its timestamp is '20160226-091125-203191'
def currentTimestamp():
    currentDatetime = datetime.datetime.now()
    datetimeStr = currentDatetime.__str__().split(" ")
    dateStr = ''.join(datetimeStr[0].split('-'))
    timeStr_list = datetimeStr[1].split('.')
    timeStr1 = ''.join(timeStr_list[0].split(':'))
    timeStr2 = timeStr_list[1]
    # timeStr = ''.join(datetimeStr[1].split('.')[0].split(':')[:2])
    return '-'.join([dateStr, timeStr1, timeStr2])


def predict(test_data, kernel, gstruct, fns_est):
    parameters = Parameters(gstruct, kernel, [kernel] * gstruct.size)
    parameterizedTestData = ParameterizedData(test_data, parameters)
    test_ghat_list = [fns_est[i](parameterizedTestData.getXFromGroup(i))
                      for i in range(1, gstruct.size + 1)]
    test_ghat = sum(test_ghat_list) + fns_est[0]  # with intercept
    return test_ghat


def predictionError(g, ghat):
    return sum((g - ghat) ** 2) / len(g)
