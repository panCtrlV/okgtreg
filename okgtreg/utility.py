__author__ = 'panc'

'''
Utility functions
'''


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


if __name__ == "__main__":
    print partitions(set(range(1, 7)))
    allpartitions = list(partitions(set(range(1, 7))))
    allpartitions = [tuple(list(item) for item in group) for group in allpartitions]
