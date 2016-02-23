from okgtreg.Kernel import Kernel


class Parameters(object):
    """
    Encapsulating the model parameters: group structure and kernels
    """

    # TODO: allow variable number of x kernel functions via **kwargs
    def __init__(self, group, ykernel, xkernels):
        """
        :type group: Group
        :param group:
        :type kernels: list of Kernel objects
        :param kernels:
        :return:
        """
        if not isinstance(ykernel, Kernel):
            raise AttributeError("** The argument \"ykernel\" is not of Kernel type. **")

        if not all([ isinstance(kernel, Kernel) for kernel in xkernels ]):
            raise AttributeError("** The argument \"xkernel\" does not contain elements of Kernel type. **")

        if group.size != len(xkernels):
            raise ValueError("** Each group must be equipped with a kernel. "
                             "There are %d groups, and %d kernels. ** " % (group.size, len(xkernels)))

        self.p = group.p
        self.partition = group.partition
        self.groupSize = group.size
        self.ykernel = ykernel
        self.xkernels = xkernels
        self.group = group
