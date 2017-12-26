__author__ = 'panc'

"""
Utility functions
"""

# from okgtreg.Parameters import Parameters
# from okgtreg.Data import ParameterizedData

import datetime
import functools
import inspect
import warnings


class deprecated(object):
    """This is a decorator (a decorator factory in fact) which 
    can be used to mark functions as deprecated. It will result 
    in a warning being emitted when the function is used. This 
    decorator also allows you to give a reason message. 
    
    Ref: http://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically#40301488
    """
    def __init__(self, reason):
        if inspect.isclass(reason) or inspect.isfunction(reason):
            raise TypeError("Reason for deprecation must be supplied")
        self.reason = reason

    def __call__(self, cls_or_func):
        if inspect.isfunction(cls_or_func):
            if hasattr(cls_or_func, 'func_code'):
                _code = cls_or_func.func_code
            else:
                _code = cls_or_func.__code__
            fmt = "Call to deprecated function or method {name} ({reason})."
            filename = _code.co_filename
            lineno = _code.co_firstlineno + 1

        elif inspect.isclass(cls_or_func):
            fmt = "Call to deprecated class {name} ({reason})."
            filename = cls_or_func.__module__
            lineno = 1

        else:
            raise TypeError(type(cls_or_func))

        msg = fmt.format(name=cls_or_func.__name__, reason=self.reason)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn_explicit(msg, category=DeprecationWarning, filename=filename, lineno=lineno)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return cls_or_func(*args, **kwargs)

        return new_func


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


def createDirIfNotExist(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print '%s does not exists. It is created automatically.' % dirpath
        return True
    else:
        print '%s already exists.' % dirpath
        return False


# def createLogger(name, log_fpath, level=logging.DEBUG):
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     handler = logging.FileHandler(log_fpath)
#     handler.setLevel(level)
#     formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %Z')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     return logger



if __name__ == "__main__":
    print partitions(set(range(1, 7)))
    allpartitions = list(partitions(set(range(1, 7))))
    allpartitions = [tuple(list(item) for item in group) for group in allpartitions]
