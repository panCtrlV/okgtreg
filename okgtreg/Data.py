import numpy as np

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

        :type group: Group
        :param group: a group structure, either a full partition or a structure consisting
                      of a subset of the available covariates.

        :rtype: tuple(Data, Group)
        :return: grouped data, i.e. Data(y, subset of X), and
                 a group structure with covariate indices being
                 normalized.
        """
        def flattenPartition(partition):
            return [i for g in partition for i in g]

        covariateInds = flattenPartition(group.partition)
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

    def __getitem__(self, variableName):
        """
        Given a variable name, return the corresponding data as a 1d array.

        :type variableName: str
        :param variableName: variable name

        :rtype: 1d array
        :return:
        """
        if self.yname == variableName:
            return self.y
        elif variableName in self.xnames:
            return self.X[:, self.xnames.index(variableName)]
        else:
            raise ValueError("** Variable \"%s\" is not in the data set. **" % variableName)

    def __str__(self):
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
            s1 = ('\t' + self.xnames[i] + ': ') if self.xnames is not None else ('\tX%d: ' % (i+1,))
            if self.n > 6:
                s2 = ', '.join(["%.02f" % val for val in self.X[:3, i]]) + \
                          ' ... ' + ', '.join(["%.02f" % val for val in self.X[-3:, i]])
            else:
                s2 = ', '.join(["%.02f" % val for val in self.X[:, i]])
            s3 = str(self.X[:, i].dtype)
            xStringList.append('\t'.join([s1, s2, s3]))
        xString = '\n'.join(xStringList)

        return '\n'.join([summaryString, '\n[Response]', yString, '\n[Covariates]', xString])

    def __str__(self):
        pass

class ParameterizedData(object):
    def __init__(self, data, parameters):
        if data.p != parameters.p:
            raise ValueError("** Covariates dimensions for data and parameters are not conformable."
                             "Data has %d covariates, while parameters have %d covariates." % (data.p, parameters.p))

        self.p = data.p
        self.n = data.n
        self.y = data.y
        self.X = data.X
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

    def _stackGramsForX(self):
        grams = [kernel.gram(self.getXFromGroup(i+1)) for (i, kernel) in enumerate(self.xkernels)]
        return np.vstack(grams)

    def covarianceOperatorForX(self, returnAll=False):
        """
        Calculate the covariance for X under the given group structure.

        :type returnAll: bool
        :param returnAll: if the column stack of the gram matrices (one for each group in the group structure)
                          is returned along with the covariance operator

        :rtype: 2d array or tuple of two 2d arrays
        :return: either the covariance oeprator by itself or the covariance operator together with
                 the column stack of the gram matrices.
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
