__author__ = 'panc'

'''
Utility functions
'''
# from okgtreg.Parameters import Parameters
# from okgtreg.Data import ParameterizedData

import datetime


def partitions(set_):
    """
    Generate all possible partitions from a given set
    of objects, and return a generator.
    """
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


# def predict(test_data, kernel, gstruct, fns_est):
#     parameters = Parameters(gstruct, kernel, [kernel] * gstruct.size)
#     parameterizedTestData = ParameterizedData(test_data, parameters)
#     test_ghat_list = [fns_est[i](parameterizedTestData.getXFromGroup(i))
#                       for i in range(1, gstruct.size + 1)]
#     test_ghat = sum(test_ghat_list) + fns_est[0] # with intercept
#     return test_ghat


# def predictionError(g, ghat):
#     return sum((g - ghat)**2) / len(g)


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


if __name__ == "__main__":
    print partitions(set(range(1, 7)))
    allpartitions = list(partitions(set(range(1, 7))))
    allpartitions = [tuple(list(item) for item in group) for group in allpartitions]
