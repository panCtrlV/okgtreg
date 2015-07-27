__author__ = 'panc'

# Check if a list contains same elements
# Solution provided by kennytm on stackoverflow:
#   http://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical

def checkEqual1(iterator):
    """
    Stops as soon as a difference is found.

    Since it contains more Python code, it is less efficient when many of the items are equal in the beginning.

    :param iterator:
    :return:
    """
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(first == rest for rest in iterator)
    except StopIteration:
        return True

def checkEqual2(iterator):
    """
    Content must be hashable

    Always perform O(N) copying operations, they will take longer if most of your input will return False.

    :param iterator:
    :return:
    """
    return len(set(iterator)) <= 1

def checkEqual3(lst):
    """
    must take a sequence input, typically concrete containers like a list or tuple.

    :param lst:
    :return:
    """
    return lst[1:] == lst[:-1]