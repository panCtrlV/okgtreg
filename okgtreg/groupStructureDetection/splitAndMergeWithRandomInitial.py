"""
The functions in this module implemented step-by-step split-and-merge procedure
of group structure determination proposed by Michael. The idea is as follows.

We start with a random partition of the predictor variables. The corresponding
OKGT is fitted with R2 being recorded. Then we perform the following split and
merge operators until there is no further improvement in R2:

1. Split step:

   For each group of size > 1, completely split it into univariate groups and fit
   the resulting OKGT and record its R2.

   Compare R2 of each complete split with the R2 we started with, and locate the
   group whose split gives the largest improvement in R2. Then, from this particular
   group, randomly pick a covariate to form a uni-variate group.

2. Merge step:

   For each univariate group in the current structure, try to merge it with one of
   the other groups, regardless of the other group being univariate or multivariate.
   Then, make the merge permanent for those gives the largest improvement in R2.

Then, step 1 and 2 are repeated iteratively until no further improvement in R2.

Can be extended to including removing steps for variable selection.

Question:

1. If different random partitions, do they result in the same optimal group structure?
"""

import numpy as np

from okgtreg.Group import RandomGroup, Group
from okgtreg.OKGTReg import OKGTRegForDetermineGroupStructure
from okgtreg.Parameters import Parameters


def splitAndMergeWithRandomInitial(data, kernel, useLowRankApproximation=True, rank=10, seed=None,
                                   threshold=0., nRandomPartition=2):
    """
    with aggressive split.

    :type seed: int
    :param seed: seed for random group initialization and Nystroem method
                 of low rank approximation

    :param data:
    :param kernel:
    :param useLowRankApproximation:
    :param rank:

    :type threshold: float, >=0
    :param threshold: if the improvement of merge or split in R2 exceeds this threshold,
                      it is considered significant and the merge or split is performed.
                      See "threshold" in OKGTRegForDetermineGroupStructure's methods:
                      `optimalMerge`, `optimalSplit`, `optimalSplit2`.

    :rtype: OKGTRegForDetermineGroupStructure
    :return:
    """
    method = 'nystroem' if useLowRankApproximation else 'vanilla'

    # group0 = RandomGroup(4, [i+1 for i in range(data.p)])
    group0 = RandomGroup(nRandomPartition, nCovariates=data.p, seed=seed)
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
        print("\n**** Updated group structure: %s, %.04f. ****\n" % (okgt.getGroupStructure(), okgt.r2))

    print("\n=== Final ===")
    print(">>>> Best group structure: %s." % okgt.getGroupStructure())
    return okgt


def splitAndMergeWithRandomInitial2(data, kernel, useLowRankApproximation=True, rank=10, seed=None,
                                    nRandomPartition=2, sThreshold=0., mThreshold=0., maxSplit=1):
    """
    Less aggressive version.

    :param data:
    :param kernel:
    :param useLowRankApproximation:
    :param rank:

    :type seed: int
    :param seed: seeding both random group partition for the initial group structure and
                 Nystroem method of low rank approximation for kernel matrices.

    :type sThreshold: float, >=0
    :param sThreshold: threshold for optimal split.
                       If the improvement of split in R2 exceeds this threshold, it is
                       considered significant and the merge or split is performed.
                       See "threshold" in `OKGTRegForDetermineGroupStructure.optimalSplit`
                       and `OKGTRegForDetermineGroupStructure.optimalSplit2`.

    :type mThreshold: float, >=0
    :param mThreshold: threshold for optimal merge. Similar to sThreshold.
                       See "threshold" in `OKGTRegForDetermineGroupStructure.optimalMerge`.

    :type nRandomPartition: int
    :param nRandomPartition: number of groups in the random partition as the initial group
                             structure.

    :type maxSplit: int
    :param maxSplit: see maxSplit in OKGTReg.optimalSplit2

    :return:
    """
    method = 'nystroem' if useLowRankApproximation else 'vanilla'

    # Random group initialization
    # group0 = RandomGroup(4, nCovariates=data.p, seed=seed)
    group0 = RandomGroup(nRandomPartition, nCovariates=data.p, seed=seed)
    # group0 = Group([1, 9], [2, 6, 8], [3], [4], [5], [7])
    parameters0 = Parameters(group0, kernel, [kernel]*group0.size)
    okgt = OKGTRegForDetermineGroupStructure(data, parameters0)

    # Start group structure detection procedure
    proceed = True
    counter = 0
    while proceed:
        counter += 1
        print("\n=== %d ===" % counter)

        print "[Split]"
        okgtAfterSplit = okgt.optimalSplit2(kernel, method, rank, seed, sThreshold, maxSplit)  # less aggressive split
        print "[Merge]"
        okgtAfterMerge = okgt.optimalMerge(kernel, method, rank, seed, mThreshold)

        # Determine updated group structure
        if okgtAfterSplit.bestR2 == okgtAfterMerge.bestR2:  # no split or merge can improve fitting
            proceed = False
        elif okgtAfterSplit.bestR2 > okgtAfterMerge.bestR2:
            okgt = okgtAfterSplit
        else:
            okgt = okgtAfterMerge
        print("\n**** Updated group structure after split and merge: "
              "%s, R2 = %.04f. ****\n" % (okgt.getGroupStructure(), okgt.r2))

    print("\n=== Final ===")
    print(">>>> Best group structure: %s." % okgt.getGroupStructure())
    return okgt


if __name__=='__main__':
    import numpy as np

    from okgtreg.DataSimulator import DataSimulator
    from okgtreg.Kernel import Kernel

    # Simulate data
    n = 500
    np.random.seed(25)
    data = DataSimulator.SimData_Wang04WithInteraction2(n)
    kernel = Kernel('gaussian', sigma=0.5)

    # Aggressive split
    print("=== Aggressive split ===")
    optimalOkgt = splitAndMergeWithRandomInitial(data, kernel, seed=25)
    print optimalOkgt.getGroupStructure(), '\n'
    #
    # # -----------------
    # from okgtreg.Group import Group
    # from okgtreg.OKGTReg import OKGTReg
    #
    # trueGroup = Group([1], [2], [3], [4], [5], [6,7,8])
    # trueParameters = Parameters(trueGroup, kernel, [kernel]*trueGroup.size)
    # trueOkgt = OKGTReg(data, trueParameters)
    # fit = trueOkgt.train('nystroem', 10, 25)
    # fit['r2']
    # # -----------------

    # Less aggressive split
    print "=== Less aggressive split ==="
    optimalOkgt2 = splitAndMergeWithRandomInitial2(data, kernel, seed=25)
    print optimalOkgt2.getGroupStructure(), '\n'
