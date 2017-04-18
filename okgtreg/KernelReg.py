__author__ = 'panc'

'''
In OKGT, if the response is not transformed or its transformation
is known, then we just need to find the transformations for the
covariates by using ridge-regression type algorithm.

For example, suppose the group structure is give with 6 groups and
let K = K1 + ... + K6 and the sample size is n, then we need to solve
the following regularized minimization problem:

    min_f 1/2n*sum(yi - f(xi))^2 + lambda/2*||f||_K     (1)

where f = f1 + ... + f6 and ||.||_K is the direct sum RKHS norm.

The ||.||_K and the norms of each individual RKHS Kj is connected
through:

    ||f||_K = min_{f1 in H_K1, ..., f6 in H_K6} ||f1||_K1 + ... + ||f6||_K6

So regularizing ||f||_K in (1) is equivalent to regularizing the min of
the sum.
'''

from okgtreg.Parameters import Parameters


class KernelReg(object):
    def __init__(self, y, x, kernel, group, eps=1e-6):
        self.y = y
        self.x = x
        self.kernel = kernel  # same kernel for all groups
        self.group = group
        self.eps = eps

    def train_additive(self):
        # train the regularized LS using the additive kernel
        n = self.y.shape[0]  # sample size
        l = self.group.size  # number of groups

        pass
