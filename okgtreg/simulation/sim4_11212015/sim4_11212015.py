# The following two lines show where the file I am executing from.
# Details can be found at:
#   http://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory

# import os
# print os.path.dirname(os.path.realpath(__file__))

# By adding the following two lines, the simulation script can be
# called from anywhere.
# Details can be found at:
#   http://stackoverflow.com/questions/26849832/python-imports-relative-path
import sys
sys.path.append('../okgtreg')


# Handling command-line arguments. Details can be found at:
#   http://www.diveintopython.net/scripts_and_streams/command_line_arguments.html
arg = sys.argv[1]  #


from okgtreg.Parameters import *
from okgtreg.okgtreg import *
# from ..Parameters import *
# from ..okgtreg import *

# import threading
# import time
# import logging
import multiprocessing
import collections
import pickle


"""
What is the effect of the regularization coefficient $\lambda$ on OKGT fitting
under different group structures?

In order to conduct investigation regarding this question, we simulate data from
models with different group structure specifications. For each model, the estimation
is done by using different values of $\lambda$. The following are the breakdown of
the steps:

1. Specify a model with a specific group structure
2. Fix a grid of values for $\lambda$ to be used for estimation
3. Report the R^2

# Specifying models:

We use the models with the following group structures:

1. [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
2. [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]
3. [1, 2, 3], [4, 5, 6], [7, 8, 9], [10]
4. [1, 2, 3], [4, 5, 6], [7, 8, 9, 10]
5. [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
6. [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Regularization coefficient $lambda$:

For each of the above model, OKGT is estiamted by using the following values of $\log(\lambda)$:

     -1, -2, ..., -30

so the corresponding values of $lambda$ are in the range:

    0.367879441, 0.135335283, ..., 9.35762297e-14

For each combination of group structure and $lambda$, we simulation for 100 times.
Given the same model, the data used for different $\lambda$ are the same to ensure
comparability.
"""


#########################
# Define data simulator #
#########################
# Each group is transformed by sin(). If there are multiple covariates in a group,
# the transformation is applied on their product. The response is log-transformed.
def simData_sim4(n, group):
    """
    Given a group structure, simulate the data set according to the following rule:
    1) Each group is transformed by sin(x) or sin() or sin(x1 * x2 * ... * x_p),
       where the former is for a univariate group while the latter is for a
       multi-variate group
    2) All the groups are added after transformations
    3) The response y is obtained by a power transformation of the sum from (2),
       currently we use power 3.

    :type n: int
    :param n:

    :type group: Group
    :param group:

    :rtype: Data
    :return: y and X as a Data object
    """
    p = group.p
    X = np.random.normal(0., 1., n*p).reshape((n, p))

    def transformOneGroup(g):
        """

        :type g: list
        :param g: list of covariate indices for a single group

        :rtype: 1d array
        :return: transformed covariates for one group, i.e. sin(x) or sin(x1 * x2 * ... * x_p)
        """
        g = [i-1 for i in g]
        if len(g) == 1:
            return np.sin(X[:, g[0]])
        else:
            return np.sin(X[:, g].prod(1))

    tX = np.vstack([transformOneGroup(g) for g in group.partition]).T  # transformed groups
    y = tX.sum(1) ** 3
    return Data(y, X)

#####################
# Sub-class Process #
#####################
# Estimating OKGT for a group & eps combination
class SimProcess(multiprocessing.Process):
    def __init__(self, queue, data, group, eps):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.data = data
        self.group = group
        self.eps = eps

    def run(self):
        parameters = Parameters(self.group, kernel, [kernel]*self.group.size)
        okgt = OKGTReg(self.data, parameters, eps)
        res = okgt.train_Vanilla()
        self.queue.put({np.log(eps) : res['r2']})
        print("%s - %s - group: %s - log(eps): %d" % (self.pid, self.name, self.group.name, np.log(self.eps)))

# During each simulation, same data is used.
# OKGT estimation for one eps is carried out in a child process.
# The estimation result from one process is saved in a Queue object without order.
# The following function convert a Queue object into a OrderedDict object.
def queueToOrderedDict(queue):
    """
    Convert a queue of okgt results from different processes into a OrderedDict object.
    Standard Python dictionaries are unordered. Details can be found at:
        http://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key

    :type queue: multiprocessing.Queue
    :param queue:
    :return:
    """
    d = dict()
    while not queue.empty():
        d.update(queue.get())
    od = collections.OrderedDict(sorted(d.items()))
    return od


if __name__ == '__main__':
    # Fix models / group structures
    group1 = Group([1], [2], [3], [4], [5], [6], [7], [8], [9], [10], name='group1')
    group2 = Group([1, 2], [3, 4], [5, 6], [7, 8], [9, 10], name='group2')
    group3 = Group([1, 2, 3], [4, 5, 6], [7, 8, 9], [10], name='group3')
    group4 = Group([1, 2, 3], [4, 5, 6], [7, 8, 9, 10], name='group4')
    group5 = Group([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], name='group5')
    group6 = Group([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='group6')

    # Value grid for regularization coefficient on log-scale
    nEps = 30  # mean has 16 cores, leave 2 for the other people
    lnEpsGrid = (np.arange(nEps) + 1) * (-1.)
    epsGrid = np.exp(lnEpsGrid)

    # Fixed kernel for all groups
    kernel = Kernel("gaussian", sigma=0.5)

    # Create a list to hold running Processor objects
    processes = []

    # Current group structure
    # g = group6
    if arg == 'group1':
        g = group1
    elif arg == 'group2':
        g = group2
    elif arg == 'group3':
        g = group3
    elif arg == 'group4':
        g = group4
    elif arg == 'group5':
        g = group5
    elif arg == 'group6':
        g = group6
    else:
        raise ValueError("** Group structure name is not valid. **")

    # Sample size
    nSample = 500

    # Number of simulations for each group & eps combination
    nSim = 100

    # Seeds
    seeds = np.arange(nSim)

    # Track progress
    counter = 0

    # List to hold results
    resList = []

    while counter < nSim:
        seed = seeds[counter]
        print("=== counter: %d - group: %s - seed: %d ===" % (counter, g.name, seed))

        # Process needs a Queue() to receive the results
        q = multiprocessing.Queue()

        # Simulate one data set for all eps
        np.random.seed(seed)
        data = simData_sim4(nSample, g)

        # For each eps, run a process
        for eps in epsGrid:
            proc = SimProcess(q, data, g, eps)
            processes.append(proc)
            proc.start()

        for p in processes:
            p.join()

        resList.append(queueToOrderedDict(q))

        counter += 1

    filename = 'sim4-' + g.name + '.pkl'
    pickle.dump(resList, open(filename, 'wb'))


# res = pickle.load(open("group6.pkl"))
# [d[-30] for d in res]




###################################
# Processing results and plotting #
###################################
import pickle
import matplotlib.pyplot as plt

filenames = ['sim4-group' + str(i) + '.pkl' for i in [2,3,4,5,6]]
folder = '/Users/panc25/sshfs_map/'

# Plot mean curve for each group structure
for filename in filenames:
    # filename = 'sim4-group4.pkl'

    resList = pickle.load(open(folder+filename, 'r'))

    x = resList[0].keys()
    y = np.vstack([res.values() for res in resList]).mean(0)

    # Curve: log-eps vs average R2
    plt.plot(x, y)

plt.title(r'Average $R^2$ vs $\ln(\epsilon)$ for Six Group Structures')
plt.xlabel(r'$\ln(\epsilon)$')
plt.ylabel(r'$R^2$')
# plt.legend(['group' + str(i) for i in [2,3,4,5,6]], loc=0)
groupObjs = [group2, group3, group4, group5, group6]
plt.legend(['group' + str(i) + ': ' + str(groupObjs[i-2]) for i in [2,3,4,5,6] ], loc=0)

# Plot box plot for each group structure
i = 3  # group index
plt.figure(i)
filename = filenames[i-2]

resList = pickle.load(open(folder+filename, 'r'))
x = resList[0].keys()
labels = [int(val) for val in x]
r2 = np.vstack([res.values() for res in resList])
plt.boxplot(r2, labels=labels)
plt.title(r'Boxplot of $R^2$ at different value of $\ln(\epsilon)$ for group' + str(i))
plt.xlabel(r'$\ln(\epsilon)$')
plt.ylabel(r'$R^2$')
plt.show()


"""Simulation result:
** The conclusions here are based on the simulation results from Group 2-6. The simulation
for Group 1 is still running on mean by the time this report is written. **

Two types of plots are made for the simulation results:

    1) Mean curve of estiamted R2 vs ln(eps) for each group structure;

    2) Boxplot of estimated R2 vs ln(eps) for each group structure.

The former is used to investigate the average effect of regularization parameter on R2, while
the latter is used to show the estimation variability for a given model at different level of
the regularization coefficient.

The followings are the observational conclusions:

1. From the mean curve plot, it seems that the performance of OKGT, in terms of R2 estimation,
   is quite stable when "eps" is small until ln(eps) = -13 (i.e. eps = 2.260329e-06). After that, the
   estimation deteriorates as eps becomes larger. This justifies the usage of 1e-6 as the default
   value for eps in our OKGT implementation.

   ** The performance of OKGT estimation may also depend on sample size. But this is beyond
   the scope of the current simulation. **

2. From the mean curve plot, it can also be noticed that the degeneration speed varies among different
   models. Group 6 has only one group, which displays the fastest degeneration. Group 2 has the
   largest number of groups, whose degeneration speed is the slowest. So it seems that the effect
   of eps is more prominent when there are fewer groups.

3. From the boxplots, we can see the variance of R2 estimates increases when ln(eps) exceed -13.
   This in general the case, however, for Group 2, R2 estimates show greater variability at ln(eps) = -30.
"""