import numpy as np
import traceback, sys
import collections

from okgtreg.Group import Group


"""
Classes for data
"""

class Data(object):
    """
    Encapsulating data including: response, covariate matrix, sample size,
    covariate dimension. Also included is a method to create a grouped data.
    """
    def __init__(self, y, X):
        """

        :type y: 1d array
        :param y:

        :type X: 2d array
        :param X:

        :return:
        """
        self.y = y
        self.X = X
        self.n = len(y)  # sample size
        self.p = X.shape[1]  # covariate dimension
        self.yname = None  # response name
        self.xnames = None  # covariate names

    def getGroupedData(self, group):
        """
        Given a group structure, either a full partition or a structure consisting of
        a subset of the available covariates, return the corresponding data set and
        the group structure after normalizing the covariate indices.

        The purpose of index normalization is to make the grouped data set look as if
        it is a standalone data set for the subsequent operations. To implement the
        normalization, the smallest covariate index in the given group is shifted to
        be 1 and all the other indices are adjusted accordingly, so that the relative
        differences are not changed. For example:

            ([3, 5], [4], [7, 8])] -> ([1, 3], [2], [5, 6])

        :type group: Group or list
        :param group: a group structure, either a full partition or a structure consisting
                      of a subset of the available covariates.

        :rtype: tuple(Data, Group)
        :return: grouped data, i.e. Data(y, subset of X), and
                 a group structure with covariate indices being
                 normalized.
        """
        def flattenPartition(partition):
            return [i for g in partition for i in g]

        # If the input group is a list, then construct a Group object
        # from the list
        if isinstance(group, list):
            group = Group(*tuple(group))

        covariateInds = flattenPartition(group.getPartitions())
        subX = self.X[:, np.array(covariateInds)-1]
        y = self.y
        subData = Data(y, subX)

        def normalizeCovariateIndicesForPartition(partition):
            # partition is a tuple of lists
            lens = [len(g) for g in partition]
            offsets = np.array(lens).cumsum() - lens + 1
            rawOrders = [np.arange(l) for l in lens]
            return tuple(list(rawOrders[i] + offsets[i]) for i in xrange(len(partition)))

        normalizedPartition = normalizeCovariateIndicesForPartition(group.partition)
        normalizedGroup = Group(*normalizedPartition)
        return subData, normalizedGroup

    def setXNames(self, names):
        """
        Assign names for covariates in the data set. Each covariate must have one unique name.
        The order of the names given is the same as the order of the columns in the data set.

        :type names: list of strings
        :param names:

        :rtype: None
        :return:
        """
        if len(names) != self.p:
            raise ValueError("** The number of names (%d) is different from "
                             "the number of covariates (%d). **" % (len(names), self.p))

        if not isinstance(names, list):
            try:
                names = list(names)
            except TypeError:
                print "** Failed to convert \"names\" to a list. **"

        self.xnames = names
        return

    def setYName(self, name):
        self.yname = name
        return

    def __getitem__(self, key):
        """
        Given a variable name, return the corresponding data as a 1d array.

        :type key: str or int or list of str or list of int
        :param key: If key is given as a string, then it is considered as a variable name,
                    either the response variable or one fo the covariates.

                    If key is given as an integer, then it is considered as an index at which
                    position a variable is retrieved. If key is 0, the response is returned. If
                    key is a positive integer, the corresponding covariate is returned.

                    If key is given as a list of strings, each string is considered as a variable name,
                    then the observations for the corresponding variables are returned as a 2d array.

                    If key is given as a list of integers, each integer is considered as the index
                    for a variable, where 0 is corresponding to the response, and a positive integer
                    is corresponding to a covariate.

        :rtype: 1d array or 2d array
        :return:
        """
        if isinstance(key, str):
            try:
                if key == self.yname:
                    return self.y
                elif key in self.xnames:
                    return self.X[:, self.xnames.index(key)]
                else:
                    raise ValueError("** Variable \"%s\" is not in the data set. **" % key)
            except TypeError:
                # Details about "Print or retrieve a stack traceback" can be found at:
                #   https://docs.python.org/2/library/traceback.html
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("** Variable names are not assigned. **")
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
        elif isinstance(key, int):
            if key == 0:
                return self.y
            elif key > 0:
                try:
                    return self.X[:, key-1]
                except IndexError:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    print("** Index %d is out of bounds for total number of covariates "
                          "(%d). **" % (key, self.p))
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
            else:
                raise IndexError("** Index %d is out of bounds. **" % key)
        elif isinstance(key, collections.Iterable):
            if all(isinstance(k, str) for k in key):
                # get columns by list of names
                return np.vstack([self.__getitem__(k) for k in key]).T  # recursion
            elif all(isinstance(k, int) for k in key):
                # get columns by list of indices
                return np.vstack([self.__getitem__(k) for k in key]).T  # recursion
        elif isinstance(key, slice):
            return Data(self.y[key], self.X[key, :])
        else:
            raise IndexError("** Index type %s is not recognized. **" % type(key))

    def __add__(self, other):
        """
        Combine two conformable Data objects

        :type other: Data
        :param other:
        :return:
        """
        # TODO: check self and other are conformable
        y = np.hstack([self.y, other.y])  # y is univariate
        x = np.vstack([self.X, other.X])
        return Data(y, x)

    def __str__(self):
        # Used for `print Data`

        # Summary string
        ss1 = 'Sample size: %d' % self.n
        ss2 = 'Number of covariates: %d' % self.p
        summaryString = '\n'.join([ss1, ss2])

        # Prepare print string for response
        yString1 = ('\t' + self.yname + ': ') if self.yname is not None else '\tY: '
        if self.n > 6:
            yString2 = ', '.join(["%.02f" % val for val in self.y[:3]]) + \
                       ' ... ' + ', '.join(["%.02f" % val for val in self.y[-3:]])
        else:
            yString2 = ', '.join(["%.02f" % x for x in self.y])
        yString3 = str(self.y.dtype)
        yString = '\t'.join([yString1, yString2, yString3])

        # Prepare print string for covariates
        xStringList = []
        for i in xrange(self.p):
            s1 = ('\t' + str(i+1) + '. ' + self.xnames[i] + ': ') if \
                self.xnames is not None else ('\tX%d: ' % (i+1,))

            if self.n > 6:
                s2 = ', '.join(["%.02f" % val for val in self.X[:3, i]]) + \
                          ' ... ' + ', '.join(["%.02f" % val for val in self.X[-3:, i]])
            else:
                s2 = ', '.join(["%.02f" % val for val in self.X[:, i]])
            s3 = str(self.X[:, i].dtype)
            xStringList.append('\t'.join([s1, s2, s3]))
        xString = '\n'.join(xStringList)

        return '\n'.join([summaryString, '\n[Response]', yString, '\n[Covariates]', xString])

    def __repr__(self):
        # Used for just calling `Data`
        return self.__str__()


    def getGroupedNames(self, group):
        if self.xnames is not None:
            return [[self.xnames[i-1] for i in part] for part in group.getPartitions()]
        else:
            raise ValueError("** Covariate names are not assigned to the data. **")


class ParameterizedData(object):
    def __init__(self, data, parameters):
        """

        :type data: Data
        :param data:

        :type parameters: Parameters
        :param parameters:

        :return:
        """
        if data.p != parameters.p:
            raise ValueError("** Covariates dimensions for data and parameters "
                             "are not conformable. Data has %d covariates, while "
                             "parameters have %d covariates." % (data.p, parameters.p))

        self.p = data.p
        self.n = data.n
        self.y = data.y
        self.X = data.X
        self.yname = data.yname
        self.xnames = data.xnames
        self.partition = parameters.partition
        self.groupSize = parameters.groupSize
        self.ykernel = parameters.ykernel
        self.xkernels = parameters.xkernels
        self.group = parameters.group

    def getXFromGroup(self, groupNumber = None):
        """
        Return the data sub-matrix of the covariate matrix, corresponding to
        the given group number. The group number starts from 1.

        :type groupNumber: None or int
        :param groupNumber: the group number whose covariate sub-matrix is returned. If None, by default,
                            the whole covariate matrix is returned.

        :rtype: 1d or 2d ndarray
        :return: covariate sub-matrix
        """
        if groupNumber is None:
            return self.X
        else:
            if groupNumber <= 0 or groupNumber > self.groupSize:
                raise ValueError("** 'groupNumber' is out of bound. **")

            cols = np.array(self.partition[groupNumber - 1])  # group number start from 1
            return self.X[:, cols - 1]

    def _getGramsForX(self, centered=True):
        """
        Contruct the gram matrix (centered by default) for each
        covariate group, all the gram matrices are returned in a list.

        :rtype: list of 2d arrays
        :return: gram matrices (centered by default), one for each group
        """
        xgram_list = [kernel.gram(self.getXFromGroup(i + 1), centered)
                      for (i, kernel) in enumerate(self.xkernels)]
        return xgram_list

    def _stackGramsForX(self):
        grams = self._getGramsForX()
        return np.vstack(grams)

    def covarianceOperatorForX(self, returnAll=False):
        """
        Calculate the covariance for X under the given group structure.

        :type returnAll: bool
        :param returnAll: if the column stack of the gram matrices (one
                          for each group in the group structure) is returned
                          along with the covariance operator

        :rtype: 2d array or tuple of two 2d arrays
        :return: either the covariance oeprator by itself or the covariance
                 operator together with the column stack of the gram matrices.
        """
        vstackedGrams = self._stackGramsForX()
        cov = vstackedGrams.dot(vstackedGrams.T) / self.n
        if returnAll:
            return cov, vstackedGrams
        else:
            return cov

    def covarianceOperatorForY(self, returnAll=False):
        yGram = self.ykernel.gram(self.y[:, np.newaxis])
        cov = yGram.dot(yGram.T) / self.n
        if returnAll:
            return cov, yGram
        else:
            return cov

    def crossCovarianceOperator(self):
        # return R_yx: H_x -> H_y
        yGram = self.ykernel.gram(self.y[:, np.newaxis])  # need kernel for y
        xStackedGrams = self._stackGramsForX()
        crossCov = yGram.dot(xStackedGrams.T) / self.n
        return crossCov

    def getGroupedNames(self):
        if self.xnames is not None:
            return [[self.xnames[i-1] for i in part] for part in self.group.getPartitions()]
        else:
            raise ValueError("** Covariate names are not assigned to the data. **")


# The following ParameterizedDataWithAdditiveKernel class is a
# subclass of ParameterizedData.
#
# In the subclass, a new function
# `_addGramsForX` is created to evaluate the gram matrix for X as
# a sum of the individual gram matrices.
#
# In addition, the two functions `covarianceOperatorForX` and
# `crossCovarianceOperator` are overridden to accommodate the
# new structure of $K_X$.
class ParameterizedDataWithAdditiveKernel(ParameterizedData):
    def _addGramsForX(self, centered=True):
        """
        Add component gram matrices (each of size n*n) together
        instead of stack them. The result is still a n*n matrix.

        :return:
        """
        grams = self._getGramsForX(centered)
        # return reduce(lambda x, y: x + y, grams)
        return sum(grams)

    def covarianceOperatorForX(self, returnAll=False):
        """
        Covariance operator by multiplying the gram matrix with
        itself, where the gram matrix is constructed from the
        additive kernel as a sum of component gram matrices.

        :param self:
        :return:
        """
        grams = self._getGramsForX()
        if returnAll:
            # Gx = reduce(lambda x, y: x + y, grams)
            Gx = sum(grams)
            Rxx = Gx.dot(Gx.T) / self.n
            return Rxx, Gx, grams
        else:
            # Gx = reduce(lambda x, y: x + y, grams)
            Gx = sum(grams)
            Rxx = Gx.dot(Gx.T) / self.n
            return Gx

    def crossCovarianceOperator(self):
        """
        # return KyKx as R_yx: H_x -> H_y, where Kx is the additive gram

        xStackedGrams = self._stackGramsForX()
        crossCov = yGram.dot(xStackedGrams.T) / self.n
        return crossCov

        :param self:
        :return:
        """
        Ky = self.ykernel.gram(self.y[:, np.newaxis])  # need kernel for y
        Kx = self._addGramsForX()
        Ryx = Ky.dot(Kx.T) / self.n
        return Ryx


if __name__ == '__main__':
    from okgtreg.DataSimulator import DataSimulator
    from okgtreg.Parameters import Parameters
    from okgtreg.Kernel import Kernel

    data, group = DataSimulator.SimData_Wang04WithInteraction(500)
    kernel = Kernel('gaussian', sigma=0.5)
    parameters = Parameters(group, kernel, [kernel] * group.size)
    parametersData = ParameterizedDataWithAdditiveKernel(data, parameters)

    Rxx, Kx = parametersData.covarianceOperatorForX(True)
    print "Rxx shape: ", Rxx.shape
    print "Kx shape: ", Kx.shape

    Ryx = parametersData.crossCovarianceOperator()
    print "Ryx shape: ", Ryx.shape
