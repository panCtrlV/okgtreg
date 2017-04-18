__author__ = 'panc'

'''
Fitting group structures selected from CV

This the in sequal of "housing_backward_cvalidate_analyze.py"
'''
import sys
import numpy as np

from okgtreg.Group import Group
from okgtreg.Kernel import Kernel
from okgtreg.utility import currentTimestamp
from okgtreg.DataUtils import readHousingData
from okgtreg.OKGTReg import OKGTReg2

# Selected group structures
gstruct_tupleList = [([1, 12], [2, 8], [3, 7], [4, 9], [5], [6, 11], [10, 13]),
                     ([1, 2], [3, 4], [5, 8], [6, 9], [7, 13], [10, 12], [11]),
                     ([1, 6], [2, 11], [3], [4, 5], [7, 13], [8, 9], [10, 12]),
                     ([1, 6], [2, 11], [3], [4, 9], [5, 8], [7, 13], [10, 12]),
                     ([1, 6], [2, 8], [3, 13], [4, 9], [5], [7, 11], [10, 12]),
                     ([1, 6], [2, 8], [3, 4], [5], [7, 13], [9, 11], [10, 12]),
                     ([1, 6], [2, 8], [3, 9], [4, 5], [7, 13], [10, 12], [11]),
                     ([1, 6], [2], [3, 4], [5, 8], [7, 13], [9, 11], [10, 12])]

gstruct_list = [Group(*gstruct_tuple) for gstruct_tuple in gstruct_tupleList]

# # Parse command line arguments
# args = sys.argv
# gstruct_id = int(args[1])  # mu_id 1~8


# # Current time
# timestamp = currentTimestamp()

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Read data
data = readHousingData()

############################
# Model fitting with the   #
# selected group structure #
############################
# Choose a group structure to estimate the
#   transformation functions
gstruct_id = 4
cur_gstruct = gstruct_list[gstruct_id - 1]
print "Current group structure:", cur_gstruct
okgt = OKGTReg2(data, kernel=kernel, group=cur_gstruct)
fit = okgt._train_lr(data.y)
print 'r2:', fit['r2']

####################################
# Plot functions of groups (1 ~ 7) #
####################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

## [Reference]: Remove the DOTS BORDER in a scatter plot
##      http://stackoverflow.com/questions/14325773/how-to-\
##      change-marker-border-width-and-hatch-width
##
## [Reference]: Reduce left and right MARGINS in matplotlib plot
##      http://stackoverflow.com/questions/4042192/reduce-\
##      left-and-right-margins-in-matplotlib-plot
##
## [Reference]: Change the FIGURE SIZE drawn with matplotlib
##      http://stackoverflow.com/questions/332289/how-do-you-\
##      change-the-size-of-figures-drawn-with-matplotlib
##
## [Reference]: matplotlib 3D-PLOT
##      http://matplotlib.org/examples/mplot3d/scatter3d_demo.html
##

### ===
### Plot transformations one-by-one
### ===
### f1 (NW)
fig = plt.figure(figsize=(3, 2.5))
ax = fig.add_subplot(111, projection='3d')
X = data.X[:, 0]
Y = data.X[:, 5]
Z = fit['f'][:, 0]
ax.scatter(X, Y, Z, s=3, linewidth='0')
ax.set_xlabel(data.xnames[0])
ax.set_ylabel(data.xnames[5])
# ax.set_zlabel(r'$f_%d$' % 1)
ax.set_title(r"$f$(%s, %s)" % (data.xnames[0], data.xnames[5]))

## Specify axis ticks
# crim_max = max(data.X[:, 0])
# crim_min = min(data.X[:, 0])
# print (crim_min, crim_max)
# rm_max = max(data.X[:, 5])
# rm_min = min(data.X[:, 5])
# print (rm_min, rm_max)
ax.set_xlim(0, 90)
ax.set_ylim(3, 9)
ax.set_xticks([0, 30, 60, 90])
ax.set_yticks([3, 5, 7, 9])

fig.tight_layout()
ax.view_init(elev=8, azim=-105)
fig.show()

### f2
fig = plt.figure(figsize=(3, 2.5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 1], data.X[:, 10], fit['f'][:, 1], s=3, linewidth='0')
ax.set_xlabel(data.xnames[1])
ax.set_ylabel(data.xnames[10])
ax.set_zlabel(r'$f_%d$' % 2)
ax.set_title(r"$f$(%s, %s)" % (data.xnames[1], data.xnames[10]))
fig.tight_layout()
fig.show()

### f3 (NE)
fig = plt.figure(figsize=(3, 2.5))
ax = fig.add_subplot(111)
ax.scatter(data.X[:, 2], fit['f'][:, 2], s=5, linewidth='0')
ax.set_xlabel(data.xnames[2])
# ax.set_ylabel(r'$f_%d$' % 3)
ax.set_title(r"$f$(%s)" % data.xnames[2])
fig.tight_layout()
fig.show()

### f4
fig = plt.figure(figsize=(3, 2.5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 3], data.X[:, 8], fit['f'][:, 3], s=5, linewidth='0')
ax.set_xlabel(data.xnames[3])
ax.set_ylabel(data.xnames[8])
ax.set_zlabel(r'$f_%d$' % 4)
ax.set_title(r"$f$(%s, %s)" % (data.xnames[3], data.xnames[8]))
fig.tight_layout()
fig.show()

### f5 (SW)
fig = plt.figure(figsize=(3, 2.5))
ax = fig.add_subplot(111, projection='3d')
X = data.X[:, 4]
Y = data.X[:, 7]
Z = fit['f'][:, 4]
ax.scatter(X, Y, Z, s=3, linewidth='0')
ax.set_xlabel(data.xnames[4])
ax.set_ylabel(data.xnames[7])
# ax.set_zlabel(r'$f_%d$' % 5)
ax.set_title(r"$f$(%s, %s)" % (data.xnames[4], data.xnames[7]))

# Specify ticks
print (np.min(X), np.max(X)) # (0.38500000000000001, 0.871)
print (np.min(Y), np.max(Y)) # (1.1295999999999999, 12.1265)
ax.set_xlim(0.3, 0.9)
ax.set_ylim(1, 13)
ax.set_xticks([0.3, 0.6, 0.9])
ax.set_yticks([1, 3, 5, 7, 9, 11, 13])

fig.tight_layout()
ax.view_init(elev=30, azim=-166)
fig.show()

### f6
fig = plt.figure(figsize=(3, 2.5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 6], data.X[:, 12], fit['f'][:, 5], s=5, linewidth='0')
ax.set_xlabel(data.xnames[6])
ax.set_ylabel(data.xnames[12])
ax.set_zlabel(r'$f_%d$' % 6)
ax.set_title(r"$f$(%s, %s)" % (data.xnames[6], data.xnames[12]))
fig.tight_layout()
fig.show()

### f7 (SE)
fig = plt.figure(figsize=(3, 2.5))
ax = fig.add_subplot(111, projection='3d')
X = data.X[:, 9]
Y = data.X[:, 11]
Z = fit['f'][:, 6]
ax.scatter(X, Y, Z, s=5, linewidth='0')
ax.set_xlabel(data.xnames[9])
ax.set_ylabel(data.xnames[11])
# ax.set_zlabel(r'$f_%d$' % 7)
ax.set_title(r"$f$(%s, %s)" % (data.xnames[9], data.xnames[11]))

# Specify ticks
print (np.min(X), np.max(X)) # (187.0, 711.0)
print (np.min(Y), np.max(Y)) # (0.32000000000000001, 396.89999999999998)
print (np.min(Z), np.max(Z))
ax.set_xlim(185, 715)
ax.set_ylim(0, 400)
ax.set_xticks(np.linspace(185, 715, 4).astype(int))
ax.set_yticks(np.linspace(0, 400, 4).astype(int))

fig.tight_layout()
ax.view_init(elev=20, azim=-115)
fig.show()

### ===
### Combine f1 (NW), f3 (NE), f5 (SW), f7 (SE) in a grid
### ===
fig = plt.figure(figsize=(5, 4))
# plt.tick_params(labelsize=20)
#### f1 (NW)
ax1 = fig.add_subplot(2,2,1, projection='3d')
X = data.X[:, 0]
Y = data.X[:, 5]
Z = fit['f'][:, 0]
ax1.scatter(X, Y, Z, s=2, linewidth='0')
ax1.set_xlabel(data.xnames[0], fontsize='small')
ax1.set_ylabel(data.xnames[5], fontsize='small')
ax1.set_title(r"$f$(%s, %s)" % (data.xnames[0], data.xnames[5]),
              fontsize=12)
print (min(Z), max(Z))
ax1.set_xlim(0, 90)
ax1.set_ylim(3, 9)
ax1.set_zlim(np.floor(min(Z)), np.ceil(max(Z)))
ax1.set_xticks([0, 30, 60, 90])
ax1.set_yticks([3, 6, 9])
ax1.set_zticks(np.linspace(np.floor(min(Z)), np.ceil(max(Z)), 4).astype(int))
ax1.tick_params(labelsize=10) # change the axis label size
                              # This is inspired by the reference:
                              #     http://matplotlib.1069221.n5.nabble.com/Increasing-\
                              #     font-size-in-axis-ticks-td27274.html
ax1.view_init(elev=8, azim=-105)
#### f3 (NE)
ax2 = fig.add_subplot(2,2,2)
X = data.X[:, 2]
Y = fit['f'][:, 2]
print (min(Y), max(Y))
y_lim_min = np.floor(min(Y))
y_lim_max = np.ceil(max(Y))
ax2.set_ylim(y_lim_min, y_lim_max)
ax2.set_yticks(np.linspace(y_lim_min, y_lim_max, 8).astype(int))
ax2.tick_params(labelsize=10)  # change the axis label size
                               # This is inspired by the reference:
                               #     http://matplotlib.1069221.n5.nabble.com/Increasing-\
                               #     font-size-in-axis-ticks-td27274.html
ax2.scatter(X, Y, s=4, linewidth='0')
ax2.set_xlabel(data.xnames[2], fontsize='small')
ax2.set_title(r"$f$(%s)" % data.xnames[2],
              fontsize=12)
#### f5 (SW)
ax3 = fig.add_subplot(2,2,3, projection='3d')
X = data.X[:, 4]
Y = data.X[:, 7]
Z = fit['f'][:, 4]
ax3.scatter(X, Y, Z, s=2, linewidth='0')
ax3.set_xlabel(data.xnames[4], fontsize='small')
ax3.set_ylabel(data.xnames[7], fontsize='small')
ax3.tick_params(labelsize=10) # change the axis label size
                              # This is inspired by the reference:
                              #     http://matplotlib.1069221.n5.nabble.com/Increasing-\
                              #     font-size-in-axis-ticks-td27274.html
ax3.set_title(r"$f$(%s, %s)" % (data.xnames[4], data.xnames[7]),
              fontsize=12)
##### Specify ticks
# print (np.min(X), np.max(X)) # (0.38500000000000001, 0.871)
# print (np.min(Y), np.max(Y)) # (1.1295999999999999, 12.1265)
print (min(Z), max(Z))
ax3.set_xlim(0.3, 0.9)
ax3.set_ylim(1, 13)
z_lim_min = np.floor(min(Z))
z_lim_max = np.ceil(max(Z))
ax3.set_zlim(z_lim_min, z_lim_max)
ax3.set_xticks([0.3, 0.6, 0.9])
ax3.set_yticks([1, 3, 5, 7, 9, 11, 13])
ax3.set_zticks(np.linspace(z_lim_min, z_lim_max, 4).astype(int))
ax3.view_init(elev=30, azim=-166)
#### f7 (SE)
ax4 = fig.add_subplot(2,2,4, projection='3d')
X = data.X[:, 9]
Y = data.X[:, 11]
Z = fit['f'][:, 6]
ax4.scatter(X, Y, Z, s=3, linewidth='0')
ax4.set_xlabel(data.xnames[9], fontsize='small')
ax4.set_ylabel(data.xnames[11], fontsize='small')
ax4.set_title(r"$f$(%s, %s)" % (data.xnames[9], data.xnames[11]),
              fontsize=12)
# Specify ticks
# print (np.min(X), np.max(X)) # (187.0, 711.0)
# print (np.min(Y), np.max(Y)) # (0.32000000000000001, 396.89999999999998)
# print (np.min(Z), np.max(Z))
ax4.set_xlim(185, 715)
ax4.set_ylim(0, 400)
z_lim_min = np.floor(min(Z))
z_lim_max = np.ceil(max(Z))
ax4.set_zlim(z_lim_min, z_lim_max)
ax4.set_xticks(np.linspace(185, 715, 4).astype(int))
ax4.set_yticks(np.linspace(0, 400, 3).astype(int))
ax4.set_zticks(np.linspace(z_lim_min, z_lim_max, 4).astype(int))
ax4.tick_params(labelsize=10)  # change the axis label size
                               # This is inspired by the reference:
                               #     http://matplotlib.1069221.n5.nabble.com/Increasing-\
                               #     font-size-in-axis-ticks-td27274.html
ax4.view_init(elev=20, azim=-115)

fig.suptitle("Selected Transformations", fontsize=15)
fig.tight_layout(pad=2, w_pad=0.5, h_pad=0.2)
fig.show()