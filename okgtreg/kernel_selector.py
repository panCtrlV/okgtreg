__author__ = 'panc'

"""
Define kernel functions and a selector.
"""
import numpy as np

def KernelSelector(name, **kwargs):
    """
    Kernel selector. One can choose any one of the kernels provided.

    ** Input **
        kernel: a string of kernel name
        **kwargs: parameters for different kernel functions. It is accessed as a dictionary.
            "sigma" for Gaussian kernel,
            "alpha" for Laplace kernel.
            "gamma", "degree" for Polynomial kernel.

    ** Output **
        kernel function: callable.

    ** Example **
        OKGT.KernelSelector(kernel="Gaussian", sigma=0.5)
        OKGT.KernelSelector(kernel="Laplace", alpha=0.5)
        OKGT.KernelSelector(kernel="Polynomial", slope=2., intercept=1., degree=3)
    """

    # --- RBFs ---
    def GaussianKernel(x, y):
        """
        Gaussian RBF
        """
        sigma = kwargs['sigma']
        #Input: sigma is a numercial number, x and y are col vectors of same length
        #TODO: check if x and y are matrices
        norm2 = np.power(x-y, 2).sum()
        #return np.exp(-norm2 / sigma**2)
        return np.exp(- sigma * norm2)

    def ExponentialKernel(x, y):
        '''
        Exponential kernel function: exp(-sigma * ||x - y||)
        Ref: http://crsouza.org/2010/03/kernel-functions-for-machine-learning-applications/
        '''
        sigma = kwargs['sigma']
        norm = np.sqrt( np.power(x-y, 2).sum() )
        return np.exp(- sigma * norm)

    def LaplaceKernel(x, y):
        """
        Laplace fucntion
        """
        alpha = kwargs['alpha']
        absSum = np.abs(x-y).sum()
        return np.exp(- alpha * absSum)

    def AnovaKernel(x,y):
        pass

    def RationalQuadraticKernel(x,y):
        pass

    def MultiquadricKernel(x,y):
        pass

    def InverseMultiquadricKernel(x,y):
        pass

    def CircularKernel(x,y):
        pass

    def SphericalKernel(x,y):
        pass

    def WaveKernel(x,y):
        pass

    def PowerKernel(x,y):
        pass

    def LogKernel(x,y):
        pass

    def SplineKernel(x,y):
        pass

    def BsplineKernel(x,y):
        pass

    def BesselKernel(x,y):
        pass

    def CauchyKernel(x,y):
        pass

    def ChiSquareKernel(x,y):
        pass

    def HistogramIntersectionKernel(x,y):
        pass

    def GeneralizedHistogramIntersectionKernel(x,y):
        pass

    def GeneralizedTstudentKernel(x,y):
        pass

    def BayesianKernel(x,y):
        pass

    def WaveletKernel(x,y):
        pass

    def PolynomialKernel(x, y):
        '''
        Polynomial kernel function
        Ref: http://scikit-learn.org/stable/modules/svm.html
        '''
        slope = kwargs['slope']
        intercept = kwargs['intercept']
        degree = kwargs['degree']

        if hasattr(x, "__len__") and hasattr(y, "__len__"):
            # Check if x, y are arrarys, because a scalar cannot use `.sum()`.
            # For example, `(2*3).sum()` causes an error.
            innerprod = (x * y).sum()
        else:
            innerprod = x * y
        return (slope * innerprod + intercept)**degree

    def SigmoidKernel(x,y):
        '''
        Sigmoid kernel function: tanh(a * x*y + r)
        Ref: http://scikit-learn.org/stable/modules/svm.html
        '''
        a = kwargs['a']
        r = kwargs['r']
        if hasattr(x, "__len__") and hasattr(y, "__len__"):
            # Check if x, y are arrarys, because a scalar cannot use `.sum()`.
            # For example, `(2*3).sum()` causes an error.
            innerprod = (x * y).sum()
        else:
            innerprod = x * y
        return np.tanh(a * innerprod + r)

    # Select a kernel
    if name=="Gaussian":
        if len(kwargs)==1:
            return GaussianKernel
        else:
            raise Exception("[Error] Gaussian requires one parameter \"sigma\" !")
    elif name=="Laplace":
        if len(kwargs)==1:
            return LaplaceKernel
        else:
            raise Exception("[Error] Laplace requires one parameter \"alpha\" !")
    elif name=="Polynomial":
        if len(kwargs)==3:
            return PolynomialKernel
        else:
            raise Exception("[Error] Polynomial requires three parameters \"slope\", \"intercept\", and \"degree\" !")
    elif name=="Sigmoid":
        if len(kwargs)==2:
            return SigmoidKernel
        else:
            raise Exception("[Error] Sigmoid requires two parameters \"a\" and \"r\" !")
    elif name=="Exponential":
        if len(kwargs)==1:
            return ExponentialKernel
        else:
            raise Exception("[Error] Exponential kernel requires one parameter \"sigma\" !")
    else:
        raise Exception("[Error] The kernel " + name + " is not supported !")