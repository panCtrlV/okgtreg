"""
Test if the correlation between transformations can be used to pick a covariate
to split.

1. Calculate pearson correlation between transformations in the same group after complete split;
2. Calculate the average correlation between one transformation and all the other transformations;
3. Split the one with the lowest average
"""

import os
import numpy as np
import pickle

from okgtreg import *


kernel = Kernel('gaussian', sigma=0.05)
n = 500
nSim = 1000
seeds = range(nSim)

while False:
    # Simulation
    r2 = []  # R2

    r16 = []
    r17 = []
    r67 = []

    r1_67 = []
    r6_17 = []
    r7_16 = []

    r1e = []
    r6e = []
    r7e = []

    corrWithResd = []  # correlation between every transformation and residual

    counter = 0
    while counter < nSim:
        seed = seeds[counter]
        counter += 1
        print("=== sim-%d : seed=%d ===" % (counter, seed))

        np.random.seed(seed)
        data, trueGroup = DataSimulator.SimData_Wang04WithInteraction(n)
        # print(trueGroup)

        """
        The true group structure is: ([1], [2], [3], [4], [5], [6, 7])

        To test the idea, we assume [1] and [6,7] were grouped, and
        we test if [1,6,7] should be split.

        The test group structure is: ([1], [2], [3], [4], [5], [6], [7])
        """
        testGroup = trueGroup._splitOneGroup(6)
        # print(testGroup)

        testOkgt = OKGTReg(data, kernel=kernel, group=testGroup)
        # testOkgt = OKGTReg(data, kernel=kernel, group=trueGroup)
        fit = testOkgt.train('nystroem', 10, 25)
        r2.append(fit['r2'])
        print("R2 = %.06f" % fit['r2'])

        # OKGT residuals
        resd = fit['g'].reshape((n,)) - fit['f'].sum(1)

        # Extract f1, f6, f7
        f167 = fit['f'][:, [0, 5, 6]]
        # f167

        # Correlation matrix between f1, f6, f7
        corr = np.corrcoef(f167, rowvar=0)
        r16.append(corr[0, 1])
        r17.append(corr[0, 2])
        r67.append(corr[1, 2])

        # Correlation between one transformation and sum of the the othe two
        # For example, r1_67 = Corr(f1, f6+f7)
        r1_67.append( np.corrcoef(f167[:, 0], f167[:, [1,2]].sum(1))[0,1] )
        r6_17.append( np.corrcoef(f167[:, 1], f167[:, [0,2]].sum(1))[0,1] )
        r7_16.append( np.corrcoef(f167[:, 2], f167[:, [0,1]].sum(1))[0,1] )

        # Correlation between one transformation and the residual
        r1e.append( np.corrcoef(f167[:, 0], resd)[0,1] )
        r6e.append( np.corrcoef(f167[:, 1], resd)[0,1] )
        r7e.append( np.corrcoef(f167[:, 2], resd)[0,1] )

        # Correlation between every transformations and the residuals
        corrWithResd.append( [ np.corrcoef(fit['f'][:, j], resd)[0, 1] for j in range(7) ] )
        # [ np.corrcoef(fit['f'][:, j], resd)[0, 1] for j in range(6) ]
        # [ np.corrcoef(fit['f'][:, j], fit['g'].reshape(n,))[0, 1] for j in range(6) ]

    corrWithResd = np.vstack(corrWithResd)

    # fileDir = os.path.dirname(os.path.realpath(__file__))
    fileDir = os.getcwd()
    pickle.dump((r2, r16, r17, r67, r1_67, r6_17, r7_16, r1e, r6e, r7e),
                open(fileDir + '/sim1000_f167.pkl', 'wb'))
    pickle.dump(corrWithResd, open(fileDir + '/sim1000_corrWithResd.pkl', 'wb'))


"""
We simulate data from a model with two bivariate groups.

We first fit an OKGT by assuming a fully additive structure,
then plot the correlations between each transformation and the
residual. This is similar to what we did above.
"""
# r2 = []
# corrWithResd = []  # correlations between each transformation and residuals
# corr_f67e = []  # correlations between f6+f7 and residuals
# corr_f89e = []  # correlations between f8+f9 and residuals
corr_f = []

fitGroup = Group( *tuple([[i+1] for i in range(9)]) )

counter = 0
while counter < nSim:
    seed = seeds[counter]
    counter += 1
    print("=== Wang04WithTwoBivariateGroups : Sim-%d : seed=%d === " % (counter, seed))

    np.random.seed(seed)
    data, trueGroup = DataSimulator.SimData_Wang04WithTwoBivariateGroups(n)

    # Fit OKGT
    # okgt = OKGTReg(data, kernel=kernel, group=fitGroup)
    okgt = OKGTReg(data, kernel=kernel, group=trueGroup)
    fit = okgt.train('nystroem', 10, 25)
    # fit = okgt.train()
    # r2.append(fit['r2'])
    print("\tR2 = %.06f" % fit['r2'])

    # cov = fit['f'].T.dot(fit['f']) / n
    corr = np.corrcoef(fit['f'], rowvar=0)
    eigs = np.linalg.eigh(corr)[0]
    eigs

    # OKGT residuals
    # resd = fit['g'].reshape((n,)) - fit['f'].sum(1)

    # Correlation between every transformations and the residuals
    # corrWithResd.append( [ np.corrcoef(fit['f'][:, j], resd)[0, 1] for j in range(fitGroup.p) ] )
    # corrWithResd.append( [ np.corrcoef(fit['f'][:, j], resd)[0, 1] for j in range(trueGroup.size) ] )
    # corr_f67e.append( np.corrcoef(fit['f'][:, [5, 6]].sum(1), resd)[0, 1] )
    # corr_f89e.append( np.corrcoef(fit['f'][:, [7, 8]].sum(1), resd)[0, 1] )

    # Correlation among all f's
    corr_f.append( np.corrcoef(fit['f'], rowvar=0) )

# Collection all correlations as an 2d array
# corrWithResd = np.vstack(corrWithResd)

# Save results
fileDir = os.getcwd()
# pickle.dump((corrWithResd, corr_f67e, corr_f89e),
#             open(fileDir + '/sim1000-Wang04WithTwoBivariateGroups_corrWithResd.pkl', 'wb'))
# pickle.dump(corrWithResd,
#             open(fileDir + '/sim1000-Wang04WithTwoBivariateGroups-trueGroup_corrWithResd.pkl', 'wb'))
# pickle.dump(corr_f,
#             open(fileDir + '/sim1000-Wang04WithTwoBivariateGroups-trueGroup_corrf.pkl', 'wb'))
pickle.dump(corr_f,
            open(fileDir + '/sim1000-Wang04WithTwoBivariateGroups-additiveGroup_corrf.pkl', 'wb'))

