__author__ = 'panc'


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
