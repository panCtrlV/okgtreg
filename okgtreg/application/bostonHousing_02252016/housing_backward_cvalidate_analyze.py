__author__ = 'panc'

import pickle
import glob
import re
import numpy as np

############################################
# Select the best group structure from the #
# cross validation results                 #
############################################
# Set path for the result folder
pkl_folder = "okgtreg/application/bostonHousing_02252016/housing_backward_cvalidate_lra0"
pkl_file = "housing_backward_cvalidate-mu1-alpha1-20160226-144427-001541.pkl"
file_path = pkl_folder + '/' + pkl_file

# Check/Test a single pkl file
with open(file_path, 'rb') as f:
    res_dict = pickle.load(f)

# ===
# Process all .pkl files
# ===
res_dictList = []
param_error_dict = {}
param_gslist_dict = {}
counter = 0
for file_path in glob.glob(pkl_folder + '/' + "*.pkl"):
    counter += 1
    print '[%d]' % counter, file_path
    with open(file_path, 'rb') as f:
        res_dict = pickle.load(f)
    res_dictList.append(res_dict)
    param_key_str = re.search("mu\d*-alpha\d*", file_path).group(0)
    # Reference: How can I find all matches to a regular expression
    #   http://stackoverflow.com/questions/4697882/how-can-i-find-\
    #   all-matches-to-a-regular-expression-in-python
    param_key = tuple([int(s) for s in re.findall("\d+", param_key_str)])
    # param_key = re.search("mu\d*-alpha\d*", file_path).group(0)
    param_error_dict[param_key] = res_dict['avg_test_error']
    param_gslist_dict[param_key] = set([g.__str__() for g in res_dict['train_gstructs'].values()])

# ===
# The (mu, alpha) pair that gives the smallest error
# ===
min_error = min(param_error_dict.values())
best_error_key = [k for k, v in param_error_dict.items() if v == min_error]
print best_error_key  # ['mu3-alpha2', 'mu4-alpha2', 'mu2-alpha2', 'mu1-alpha2'], [(1, 2), (2, 2), (3, 2), (4, 2)]

# ===
# List group structure and test error for each optimal (mu, alpha)
# ===
for i, j in best_error_key:
    print "===", (i, j), "==="
    pkl_file = "housing_backward_cvalidate-mu" + str(i) + "-alpha" + str(j) + "*.pkl"
    pkl_file_path = glob.glob(pkl_folder + '/' + pkl_file)[0]
    with open(pkl_file_path, 'rb') as f:
        res_dict = pickle.load(f)
    for fold_id in range(1, 11):
        print res_dict['train_gstructs'][fold_id], ":", res_dict['test_errors'][fold_id]

sorted(gs.__str__() for gs in res_dict['train_gstructs'].values())
############################################
# Latex table of optimal tuning parameters #
# and the test error (logarithm)            #
############################################
## Parameter grid
mu_size = 5
alpha_size = 10
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
alphaList = np.arange(1, alpha_size + 1)

print "{0:10} & {1:2} & {2:10} \\\\".format("mu", "alpha", "log(test error)")
for i, j in best_error_key:
    print "{0:10} & {1:2} & {2:10} \\\\".format("%.5e" % muList[i - 1], alphaList[j - 1], "%.5f" % np.log(min_error))

# # Unpickle the result for the best (mu, alpha)
# pkl_file = glob.glob(pkl_folder+"/"+"housing_backward_cvalidate-mu"+best_error_key[0]+"-alpha"+best_error_key[1]+"*.pkl")[0]
# with open(pkl_file, 'rb') as f:
#     best_res_dict = pickle.load(f)

# best_res_dict

#########################################################
# Plot the average prediction error for all (mu, alpha) #
#########################################################
# ===
# LINE PLOT
# The 50 (\mu, \alpha) pairs are arranged in increasing
# order (\my first, then \alpha) along x-axis. The corr-
# sponding average prediction errors are visualized as a
# line plot.
# ===
import matplotlib.pyplot as plt

errors = [v for k, v in sorted(param_error_dict.items())]
plt.plot(range(1, 51), np.log(errors), 'bo',
         range(1, 51), np.log(errors), 'k')  # log scale

## All group structures that gives the lowest
##   predictor error, i.e. the group structures
##   selected by the best (mu, alpha)
optimal_gstruct = reduce(lambda x, y: x | y, [param_gslist_dict[k] for k in best_error_key])
for gs in sorted(list(optimal_gstruct)):
    print gs
# ([1, 12], [2, 8], [3, 7], [4, 9], [5], [6, 11], [10, 13])
# ([1, 2], [3, 4], [5, 8], [6, 9], [7, 13], [10, 12], [11])
# ([1, 6], [2, 11], [3], [4, 5], [7, 13], [8, 9], [10, 12])
# ([1, 6], [2, 11], [3], [4, 9], [5, 8], [7, 13], [10, 12]) **
# ([1, 6], [2, 8], [3, 13], [4, 9], [5], [7, 11], [10, 12])
# ([1, 6], [2, 8], [3, 4], [5], [7, 13], [9, 11], [10, 12]) *
# ([1, 6], [2, 8], [3, 9], [4, 5], [7, 13], [10, 12], [11])
# ([1, 6], [2], [3, 4], [5, 8], [7, 13], [9, 11], [10, 12])

# ===
# 3D SURFACE PLOT
# It turns out that the line plot may cause difficulty
#   in understanding the intention of the figure.
#   Instead, we replace the line plot with a 3D surface with
#   2D projections.
# ===
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

## Set x (equal spaced in log-scale), y axis values
xgv = np.log(muList)
ygv = alphaList
[X,Y] = np.meshgrid(xgv, ygv)
errors = np.array([v for k, v in sorted(param_error_dict.items())])
log_errors = np.log(errors)
Z = log_errors.reshape(X.shape, order='F')

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, Z,
                rstride=1, cstride=1,
                # cmap=plt.cm.jet, #plt.cm.CMRmap, #plt.cm.Spectral,
                linewidth=0.5,
                # antialiased=True,
                alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='x', offset=np.min(X) - 2, cmap=cm.coolwarm, alpha=0.7)
cset = ax.contourf(X, Y, Z, zdir='y', offset=np.max(Y) + 5, cmap=cm.coolwarm, alpha=0.7)
cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z) - 3, cmap=cm.coolwarm, alpha=0.7)
ax.set_xlabel(r'$\log(\mu)$')
ax.set_xlim(np.min(X) - 2, np.max(X))
ax.set_ylabel(r'$\alpha$')
ax.set_ylim(np.min(Y), np.max(Y) + 5)
# ax.set_zlabel('Z')
ax.set_zlim(np.min(Z) - 3, 15)
ax.set_title("Average Predition Errors", fontsize=15)
ax.view_init(elev=18, azim=-50)