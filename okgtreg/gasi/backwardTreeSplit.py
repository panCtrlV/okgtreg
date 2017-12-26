"""
Backward tree split algorithm for GASI.

1. Start with all covariates.
2. Split the set of covariates into two groups such that
    the bi-partition achieves the highest R2
3. For each subset, perform step 1 and 2.

@author: panc
@time: 2017-05-07
"""

import numpy as np
from okgtreg.Group import Group
from okgtreg.Parameters import Parameters


def backwardTreeSplit(data, kernel, seed, logger=None):
    """Backward tree split algorithm (each time a group is divided into two subgroups)
    
    :type data: okgtreg.Data object
    :param data: data set.
    :type kernel: 
    :param kernel:
    :type seed: int
    :param seed: seed for randomly splitting one group into two subgroups (left and right).
    """
    covariates_lst = list( np.arange(data.p) + 1 )  # all covariates 1, ...., p
    old_group = Group(covariates_lst)
    np.random.seed(seed)
    if old_group.p > 1:
        # Randomly split old group into two subgroups (left and right)
        left_p = np.random.randint(1, old_group.p+1)
        # right_p = old_group.p - left_p
        curr_group = old_group.split(1, randomSplit=True, seed=seed, splitSize=left_p, complete_split=False)
        parameters = Parameters(curr_group, kernel, )

    pass


if __name__=='__main__':

    # Simulate data
    from okgtreg.DataSimulator import DataSimulator

    n = 1000
    DS = DataSimulator()
    data = DS.Wang04WithInteraction2_100(n)

    import numpy as np
    from okgtreg.Group import Group

    covariates_lst = list( np.arange(data.p)+1 )
    old_group = Group( covariates_lst )
    np.random.seed(25)
    left_p = np.random.randint(1, old_group.p+1)
    old_group.split(1, randomSplit=True, seed=123, splitSize=left_p, complete_split=False)
    # old_group._randomSplitOneGroup(1, 123, 3, complete_split=False)