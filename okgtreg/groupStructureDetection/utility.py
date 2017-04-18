__author__ = 'panc'


def rkhsCapacity(group, alpha):
    return sum([alpha ** len(g) for g in group.partition])
