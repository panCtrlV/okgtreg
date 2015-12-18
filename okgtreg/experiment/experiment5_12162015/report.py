import pickle
import numpy as np
import collections
import matplotlib.pyplot as plt

"""
The true group structure is:

    ([1], [2], [3], [4], [5], [6, 7]),

but we fit the OKGT with the group structure:

    ([1], [2], [3], [4], [5], [6], [7])
"""
r2, r16, r17, r67, r1_67, r6_17, r7_16, r1e, r6e, r7e = \
    pickle.load(open("okgtreg/experiment/experiment5_12162015/sim1000_f167.pkl", 'rb'))

# Histograms
# fig, (ax1, ax2, ax3) = plt.subplots(3)
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.hist(r16, 50)
ax1.set_ylabel(r"Cov($f_1$, $f_6$)")
ax2.hist(r17, 50)
ax2.set_ylabel(r"Cov($f_1$, $f_7$)")
ax3.hist(r67, 50)
ax3.set_ylabel(r"Cov($f_6$, $f_7$)")

# Boxplots
plt.boxplot([np.abs(r16), np.abs(r17), np.abs(r67)])
"""
The center of r67 seems to deviate from 0, while the centers of
the other two correlations seem to be 0.

Compare average of the absolute values of correlations:
"""
corr167 = [np.mean(np.abs(r)) for r in [r16, r17, r67]]
"""
0.036306394192229072,
0.035167691191940492,
0.039139582511452956

They are almost the same.

Frequency of each absolute correlation being largest in a simulation:
"""
corrAbs = np.abs(np.vstack([r16, r17, r67])).T
maxIndices_corrAbs = corrAbs.argmax(1)
collections.Counter(maxIndices_corrAbs)
"""
Counter({0: 340, 1: 307, 2: 353})

The counter shows the frequency for each pair-wise correlation being the largest in
each simulation. It shows that of 1000 simulations, Corr(f6, f7) is the largest most
of time. But the difference in the frequencies is not significant.

For each covariate, we can also calculate the average correlation between itself and
the other two covariates. The average is taken over the absolute values of the correlations.
This is implemented as follows:
"""
avgCorrAbs = np.vstack( [corrAbs[:, [1,2]].mean(1),
                         corrAbs[:, [0,2]].mean(1),
                         corrAbs[:, [0,1]].mean(1)] ).T
minIndices_avgCorrAbs = avgCorrAbs.argmin(1)
collections.Counter(minIndices_avgCorrAbs)
"""
Counter({0: 350, 1: 345, 2: 305})

The counter shows the frequency for each covariate having the smallest average correlation
in each simulation. For covariate Xi, its average absolute correlation between itself and the
other two covariates Xj and Xk is calculated by:

    [ abs(Corr(Xi, Xj)) + abs(Corr(Xi, Xk)) ] / 2

The results show that f7 has the smallest average absolute correlation most of time, followed
by f6, and f5.

** This is against my intuition that f5 should be less dependent on the other two transformed
covariates. **

What if we compare min max of the absolute correlations for each transformation?
"""
maxCorrAbs = np.vstack( [corrAbs[:, [1,2]].max(1),
                         corrAbs[:, [0,2]].max(1),
                         corrAbs[:, [0,1]].max(1)] ).T
minmaxIndices = maxCorrAbs.argmin(1)
collections.Counter(minmaxIndices)
"""
Counter({0: 340, 1: 307, 2: 353})

Correlation between fi and (fj+fk):
"""
x = np.vstack([r1_67, r6_17, r7_16]).T
ind = np.abs(x).argmin(1)
collections.Counter(ind)
"""
Counter({0: 258, 1: 362, 2: 380})

Correlation between transformations and residuals:
"""
# Histograms
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.hist(r1e, 50); ax1.set_ylabel(r"$f_1$")
ax2.hist(r6e, 50); ax2.set_ylabel(r"$f_6$")
ax3.hist(r7e, 50); ax3.set_ylabel(r"$f_7$")
fig.suptitle(r"Correlation between $f_1$, $f_6$, $f_7$ and residuals" + '\n' +
             r"group structure ([1],[2],[3],[4],[5],[6],[7])")

# Boxplots
fig, ax1 = plt.subplots()
ax1.boxplot([np.abs(r1e), np.abs(r6e), np.abs(r7e)])
fig.suptitle(r"Correlation between $f_1$, $f_6$, $f_7$ and residuals" + '\n' +
          r"group structure ([1],[2],[3],[4],[5],[6],[7])")
xtickNames = plt.setp(ax1, xticklabels=[r"$f_1$", r"$f_2$", r"$f_3$"])
plt.setp(xtickNames)
"""
The comparison of the histograms shows that f6 and f7 are almost non-correlated
with the residuals of the OKGT when [1,6,7] are completely separated. However,
f1 shows some negative correlation with the residuals.

Now we plot the correlation between every transformations and residuals:
"""
corrWithResd = pickle.load(open("okgtreg/experiment/experiment5_12162015/sim1000_corrWithResd.pkl", 'rb'))
corrWithResd = corrWithResd.T

# Histograms
fig, axarr = plt.subplots(7, 1, sharex=True)
for i in range(7):
    axarr[i].hist(corrWithResd[:, i], 50)
    axarr[i].set_ylabel(r"$f_" + str(i+1) + "$")

# Boxplots
plt.boxplot(corrWithResd)
"""
The transformations of the two covariates X6 and X7 which are supposed to be together shows
negligible corrections between residuals.

How about we change to a model with two bi-variate groups? So in the following the model has
the true group structure:

    ([1], [2], [3], [4], [5], [6, 7], [8, 9])

We first fit the data using the following additive structure:

    ([1], [2], [3], [4], [5], [6], [7], [8], [9])
"""
corrWithResd, corr_f67e, corr_f89e = \
    pickle.load(open("okgtreg/experiment/experiment5_12162015/"
                     "sim1000-Wang04WithTwoBivariateGroups_corrWithResd.pkl", 'rb'))

# Histograms
p = 9
fig, axarr = plt.subplots(p, 1, sharex=True)
for i in range(p):
    axarr[i].hist(corrWithResd[:, i], 50)

# Boxplots
plt.boxplot(corrWithResd)
"""
From the histograms and boxplots, we can see that the transformations of the last
four covariates which are supposed to be grouped have lower correlations with the
residuals on average than the transformations of the other covariates.

How about the correlations between f6+f7, f8+f9 and the residuals?
"""
p = 9
fig, axarr = plt.subplots(p+2, 1, sharex=True)
for i in range(p):
    axarr[i].hist(corrWithResd[:, i], 50)
axarr[i+1].hist(corr_f67e, 50)
axarr[i+2].hist(corr_f89e, 50)
"""
The correlations between f6+f7, f8+f9 and the residuals are also smaller than the
other univariate transformations.

How about fitting the true model? In the following we fit data using the true structure:

    ([1], [2], [3], [4], [5], [6, 7], [8, 9])
"""
corrWithResd = pickle.load(open("okgtreg/experiment/experiment5_12162015/"
                                "sim1000-Wang04WithTwoBivariateGroups-trueGroup_corrWithResd.pkl", 'rb'))

# Histograms
p = 7
fig, axarr = plt.subplots(p, 1, sharex=True)
fig.suptitle("Correlations between transformations and residuals\n"
             "model: Wang04 with two bi-variate groups\n"
             "group structure: ([1],[2],[3],[4],[5],[6,7],[8,9])")
for i in range(p):
    axarr[i].hist(corrWithResd[:, i], 50)
    if i < 5:
        axarr[i].set_ylabel(r"$f_" + str(i+1) + "$")
    elif i==5:
        axarr[i].set_ylabel(r"$f_" + str(i+1) + "(X_6, X_7)$")
    else:
        axarr[i].set_ylabel(r"$f_" + str(i+1) + "(X_8, X_9)$")

# Boxplots
fig, ax1 = plt.subplots()
fig.suptitle("Correlations between transformations and residuals\n"
             "model: Wang04 with two bi-variate groups\n"
             "group structure: ([1],[2],[3],[4],[5],[6,7],[8,9])")
ax1.boxplot(corrWithResd)
ax1.set_ylabel("correlation with residuals")
xtickNames = plt.setp(ax1, xticklabels=[r"$f_1$", r"$f_2$", r"$f_3$", r"$f_4$", r"$f_5$",
                                        r"$f_6(X_6, X_7)$", r"$f_7(X_8, X_9)$"])
plt.setp(xtickNames)
"""
Under the true group structure, f4 has a significant correlation with the residuals, alomost -0.9.
Other correlations are all small.

How about the correlations among the transformations under the true model?
"""
corr_f = pickle.load(open("okgtreg/experiment/experiment5_12162015/"
                          "sim1000-Wang04WithTwoBivariateGroups-trueGroup_corrf.pkl", 'rb'))

for i in range(7):
    for j in range(7):
        if i < j:
            corr_ij = [corrMtx[i,j] for corrMtx in corr_f]
            print '[', i, ',', j, ']: ', "{:.04f}".format(np.mean(corr_ij)), "{:.04f}".format(np.std(corr_ij))

# Histograms as a matrix
l = 7
fig, axarr = plt.subplots(l, l, sharex=True, sharey=True)
fig.suptitle(r"Pairwise correlation between $f$ transformations" + '\n' +
             "model: Wang04 with two bi-variate groups\n"
             "group structure: ([1],[2],[3],[4],[5],[6,7],[8,9])")
for i in xrange(l):
    for j in xrange(l):
        if i==0:
            axarr[i, j].set_title(r"$f_" + str(j+1) + "$")
        if j == 0:
                axarr[i, j].set_ylabel(r"$f_" + str(i+1) + "$")
        if i < j:
            corr_ij = [corrMtx[i,j] for corrMtx in corr_f]
            axarr[i, j].hist(corr_ij, 50)
            # draw a vertical reference line from the bottom to the top of the y axis
            axarr[i, j].axvline(0., color='r', linewidth=2)

# Boxplots as a matrix
l = 7
fig, axarr = plt.subplots(l, l, sharex=True, sharey=True)
fig.suptitle(r"Pairwise correlation between $f$ transformations" + '\n' +
             "model: Wang04 with two bi-variate groups\n"
             "group structure: ([1],[2],[3],[4],[5],[6,7],[8,9])")
for i in xrange(l):
    for j in xrange(l):
        if i==0:
            axarr[i, j].set_title(r"$f_" + str(j+1) + "$")
        if j == 0:
                axarr[i, j].set_ylabel(r"$f_" + str(i+1) + "$")
        if i < j:
            corr_ij = [corrMtx[i,j] for corrMtx in corr_f]
            axarr[i, j].boxplot(corr_ij)
"""
By fitting using the true group structure, the pair-wise correlations between transforamed
covariates are very small.

How about when the group structure are mis-specified?
"""
corr_f = pickle.load(open("okgtreg/experiment/experiment5_12162015/"
                          "sim1000-Wang04WithTwoBivariateGroups-additiveGroup_corrf.pkl", 'rb'))

corr_f_avg = reduce(lambda x,y: x+y, corr_f) / nSim
np.sort(np.unique(corr_f_avg))

for i in range(9):
    for j in range(9):
        if i < j:
            corr_ij = [corrMtx[i,j] for corrMtx in corr_f]
            print '[', i, ',', j, ']: ', "{:.04f}".format(np.mean(corr_ij)), "{:.04f}".format(np.std(corr_ij))
            # print np.mean(np.abs(corr_ij)), np.std(np.abs(corr_ij))

# Histograms as a matrix
l = 9
fig, axarr = plt.subplots(l, l, sharex=True, sharey=True)
fig.suptitle(r"Pairwise correlation between $f$ transformations" + '\n' +
             "model: Wang04 with two bi-variate groups\n"
             "group structure: ([1],[2],[3],[4],[5],[6],[7],[8],[9])")
for i in xrange(l):
    for j in xrange(l):
        if i==0:
            axarr[i, j].set_title(r"$f_" + str(j+1) + "$")
        if j == 0:
                axarr[i, j].set_ylabel(r"$f_" + str(i+1) + "$")
        if i < j:
            corr_ij = [corrMtx[i,j] for corrMtx in corr_f]
            axarr[i, j].hist(corr_ij, 50)
            # draw a vertical reference line from the bottom to the top of the y axis
            axarr[i, j].axvline(0., color='r', linewidth=2)

# Boxplots as a matrix
l = 9
fig, axarr = plt.subplots(l, l, sharex=True, sharey=True)
fig.suptitle(r"Pairwise correlation between $f$ transformations" + '\n' +
             "model: Wang04 with two bi-variate groups\n"
             "group structure: ([1],[2],[3],[4],[5],[6,7],[8,9])")
for i in xrange(l):
    for j in xrange(l):
        if i==0:
            axarr[i, j].set_title(r"$f_" + str(j+1) + "$")
        if j == 0:
                axarr[i, j].set_ylabel(r"$f_" + str(i+1) + "$")
        if i < j:
            corr_ij = [corrMtx[i,j] for corrMtx in corr_f]
            axarr[i, j].boxplot(corr_ij)