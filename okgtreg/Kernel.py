import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.kernel_approximation import Nystroem

# pairwise_distances(x, y, kernel.eval)

"""
Kernel class object contains a kernel function (callable) and related information.
It also contains some useful methods such as evaluating a gram matrix given data.
"""

class Kernel(object):
    def __init__(self, name, sigma=None, intercept=0., slope=1., degree=None, c=0):
        self.name = name
        # TODO: replace the following if...else... with a dictionary

        kernels = {'gaussian': lambda x, y: self.gaussianKernel(x, y, sigma),
                   'laplace': lambda x, y: self.laplaceKernel(x, y, sigma),
                   'exponential': lambda x, y: self.exponentialKernel(x, y, sigma),
                   'polynomial': lambda x, y: self.polynomialKernel(x, y, intercept, slope, degree),
                   'sigmoid': lambda x, y: self.sigmoidKernel(x, y, intercept, slope),
                   'linear': lambda x, y: self.linearKernel(x, y, c)}

        if name in ('gaussian', 'laplace', 'exponential'):
            if sigma is None:
                raise ValueError("** Parameter 'sigma' is not provided for %s kernel. **" % name)
            else:
                self.sigma = sigma
                self.fn = kernels[name]
        elif name == 'polynomial':
            if any(val is None for val in [intercept, slope, degree]):
                raise ValueError("** Parameters 'intercept', 'slope', and 'degree' are not all provided"
                                 "for %s kernel. **" % name)
            else:
                self.intercept = intercept
                self.slope = slope
                self.degree = degree
                self.fn = kernels[name]
        elif name == 'sigmoid':
            if any(val is None for val in [intercept, slope]):
                raise ValueError("** Parameters 'intercept' and 'slope' are not both provided"
                                 "for %s kernel. **" % name)
            else:
                self.intercept = intercept
                self.slope = slope
                self.fn = kernels[name]
        elif name == "linear":
            self.c = c
            self.fn = kernels[name]
        else:
            raise NotImplementedError("** %s kernel is not yet implemented. **" % name)

    def eval(self, x, y):
        # evaluate the kernel at two data points
        return self.fn(x, y)

    def kernelMapping(self, x):
        # return a callable as a kernel mapping for
        # the given point x, under the given kernel
        # function
        return lambda y: self.fn(x, y)

    def kernelSpan(self, x, coef):
        """
        Construct linear combination of kernel mapping
        of a given set of points (in the form of a 2d
        array, each row is a data point). The function
        returns a callable.

        :type x: 2d array
        :param x: data points, each row is a data point

        :type coef: 1d array
        :param coef: expansion loadings

        :rtype: callable
        :return: a function as a linear combination of
                 the kernel mappings.
        """
        def kspan(y):
            if y.ndim == 1:  # y is a data point
                keval = pairwise_distances(x, y.reshape(1, -1), self.eval).squeeze()
                return keval.dot(coef)
            elif y.ndim == 2:  # y is a 2d array, each row is a data point
                keval = pairwise_distances(x, y, self.eval)  # (nrow_x * nrow_y) matrix
                return keval.T.dot(coef)
            else:
                raise ValueError("** [ERROR] the shape of y is not conformable! **")
        return kspan

    def gram(self, x, centered=True):
        # x must be a 2d array
        if x.ndim != 2:
            raise ValueError("** 'x' must be a 2-d array. **")

        n = len(x)
        G = pairwise_distances(x, metric=self.fn)
        # print G
        if centered:
            I = np.identity(n)
            Ones = np.ones((n, n))
            G = (I - Ones / n).dot(G).dot(I - Ones / n) # centered Gram matrix
            return (G + G.T)/2 # numerical issue cause asymmetry
        else:
            return G

    def gram_Nystroem(self, x, nComponents, seed=None):
        """
        Nystroem approximation of the kernel matrix given data. No centering.
        This method constructs an approximate feature map for an arbitrary kernel
        using a subset of the data as basis. For more details, please refer to:

            http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem

        :type x: 2d array, with size n * p
        :param x: data matrix for the covariates belonging to the same group, associated
                  with the given matrix.

        :type nComponents: int
        :param nComponents: number of rank to retain

        :type seed: int, optional
        :param seed: Since Nystroem method constructs the matrix approximation by selecting a
                     random subset of the data, fixing the seed for the random number generator
                     will enable creating a reproducible example.

        :return: approximated kernel matrix with reduced rank, with size n * nComponents
        """
        nystroem = Nystroem(self.fn, n_components=nComponents, random_state=seed)
        return nystroem.fit_transform(x)

    @staticmethod
    def linearKernel(x, y, c=0):
        # inner product of <x,y>
        return np.sum(x * y) + c

    @staticmethod
    def gaussianKernel(x, y, sigma):
        # sigma is a numercial number,
        # x and y are vectors of same length (dimension)
        if len(x) != len(y):
            raise ValueError("** [ERROR] x and y have different dimensions! **")
        else:
            norm2 = np.power(x - y, 2).sum()
            return np.exp(- sigma * norm2)

    @staticmethod
    def laplaceKernel(x, y, sigma):
        norm = np.linalg.norm(x - y)
        return np.exp(-sigma * norm)

    @staticmethod
    def exponentialKernel(x, y, sigma):
        """
        Exponential kernel function: exp(-sigma * ||x - y||)
        Ref: http://crsouza.org/2010/03/kernel-functions-for-machine-learning-applications/
        """
        # norm = np.sqrt(np.power(x - y, 2).sum())
        norm = np.linalg.norm(x - y)
        return np.exp(-sigma * norm)

    @staticmethod
    def polynomialKernel(x, y, intercept, slope, degree):
        """
        Ref: http://scikit-learn.org/stable/modules/svm.html
        """
        if hasattr(x, "__len__") and hasattr(y, "__len__"):
            # Check if x, y are arrarys, because a scalar cannot use `.sum()`.
            # For example, `(2*3).sum()` causes an error.
            innerprod = (x * y).sum()
        else:
            innerprod = x * y
        return (slope * innerprod + intercept) ** degree

    @staticmethod
    def sigmoidKernel(x, y, intercept, slope):
        """
        Sigmoid kernel function: tanh(a * x * y + r)
        Ref: http://scikit-learn.org/stable/modules/svm.html
        """
        if hasattr(x, "__len__") and hasattr(y, "__len__"):
            innerprod = (x * y).sum()
        else:
            innerprod = x * y
        return np.tanh(slope * innerprod + intercept)

    def __str__(self):
        return "kernel: " + str(self.__dict__)


# Class for additive kernel
class AdditiveKernel(object):
    def __init__(self, kernelList):
        self.kernelList = kernelList  # list of Kernel objects

        def addingKernels(x, y):
            '''
            Constructing additive kernels by adding up
            individual kernel functions. In order to evaluate
            the additive kernel functions, the user is
            responsible to prepare the input as list of
            1d numpy arrays.

            :param x:
            :param y:
            :return:
            '''
            pass

    def kernelMapping(self):
        pass

    def gram(self):
        pass
