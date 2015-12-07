"""
This simulation illustrate step-by-step the procedure of group structure determination
proposed by Michael.

We start with a random partition of the predictor variables. The corresponding
OKGT is fitted with R2 being recorded. Then we perform the following split and
merge operators until there is no further improvement in R2:

1. For each group of size > 1, split it into individual covariates and fit
   the resulting OKGT and record its R2.

   Compare R2 under each scenario in (2) with the R2 we started with. Pick the
   the scenario with gives the largest improvement in R2 and randomly pick a
   covariate from the group to form a uni-variate group.

2. For each univariate group in the current structure, try to merge its covariate
   with one of the other groups, regardless of the other group being univariate or
   multivariate.

Then, step 1 and 2 are repeated iteratively until no further improvement in R2.

Can be extended to include removing steps for variable selection.

Question:

1. If different random partitions, do they result in the same optimal group structure?
"""

import numpy as np

from okgtreg.Group import RandomGroup
from okgtreg.OKGTReg import OKGTRegForDetermineGroupStructure
from okgtreg.Parameters import Parameters


def splitAndMergeWithRandomInitial(seed, data, kernel, useLowRankApproximation=True, rank=10):
    """

    :type seed: int
    :param seed: seed for Nystroem method of low rank approximation

    :param data:
    :param kernel:
    :param useLowRankApproximation:
    :param rank:
    :return:
    """
    method = 'nystroem' if useLowRankApproximation else 'vanilla'

    np.random.seed(seed)
    group0 = RandomGroup(4, [i+1 for i in range(data.p)])
    parameters0 = Parameters(group0, kernel, [kernel]*group0.size)
    okgt = OKGTRegForDetermineGroupStructure(data, parameters0)

    proceed = True
    counter = 0
    while proceed:
        counter += 1
        print("\n=== %d ===" % counter)

        print "[Split]"
        okgtAfterSplit = okgt.optimalSplit(kernel, method, rank, seed)
        print "[Merge]"
        okgtAfterMerge = okgt.optimalMerge(kernel, method, rank, seed)

        if okgtAfterSplit.r2 == okgtAfterMerge.r2:
            proceed = False
        elif okgtAfterSplit.r2 > okgtAfterMerge.r2:
            okgt = okgtAfterSplit
        else:
            okgt = okgtAfterMerge

        print("\n** Updated group structure: %s, %.04f. **" % (okgt.getGroupStructure(), okgt.r2))

    print("** Best group structure: %s. **" % okgt.getGroupStructure())
    return okgt


if __name__=='__main__':
    from okgtreg.DataSimulator import DataSimulator
    from okgtreg.Kernel import Kernel


    n = 500

    np.random.seed(25)
    data = DataSimulator.SimData_Wang04WithInteraction2(n)
    kernel = Kernel('gaussian', sigma=0.5)

    optimalOkgt = splitAndMergeWithRandomInitial(25, data, kernel)
    optimalGroupStructure = optimalOkgt.getGroupStructure()
    print optimalGroupStructure
