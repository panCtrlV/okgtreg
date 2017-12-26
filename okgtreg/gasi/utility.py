__author__ = 'panc'


def rkhsCapacity(group, alpha):
    """Calculating the capacity measure for an (additive) RKHS 
    for a given group additive structure."""
    return sum([alpha ** len(g) for g in group.partition])
