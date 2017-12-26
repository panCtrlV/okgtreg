__author__ = 'panc'

'''
Analyze validate_exhaustive_bootstrap
'''

import pickle
import glob
from collections import defaultdict, OrderedDict
import numpy as np
import operator


# Unpickle .pkl files.
sim_folder = "/home/panc/research/OKGT/software/okgtreg/okgtreg/simulation/sim_02052016"
pkl_folder = "validate_exhaustive_bootstrap_model1_dataseed1"
file_name = "validate_exhaustive_bootstrap-model1-dataseed1-btseed100-201602190655.pkl"
with open(sim_folder + '/' + pkl_folder + '/' + file_name, 'rb') as f:
    res = pickle.load(f)

# Unpickle all .pkl files and
# collect bootstrap r2's
res_all_dict = defaultdict(list)
test_all_set = set()
counter = 0
for fpath in glob.glob(sim_folder + '/' + pkl_folder + '/*.pkl'):
    counter += 1
    print counter, '[processing...]', fpath
    with open(fpath, 'rb') as f:
        res = pickle.load(f)
        for k, v in res['bootstrap_r2'].iteritems():
            res_all_dict[k].append(v)
        test_all_set = test_all_set.union(res['test'].items())

# res_all_dict.items(0)
# Calculate mean r2 and std dev for each (mu, alpha)
res_all_mean_stddev_dict = {k: (np.mean(v), np.std(v)) for k, v in res_all_dict.items()}
# Calculated the mean R2 adjusted by 0.1 times the standard deviation
# for each (mu, alpha)
res_all_adjusted_dict = {k: v[0] - 0.1 * v[1] for k, v in res_all_mean_stddev_dict.items()}
res_all_adjusted_orderlist = \
    sorted(res_all_adjusted_dict.items(), key=operator.itemgetter(1), reverse=True)

pickle.dump()

# Plot
import matplotlib.pyplot as plt

with open("okgtreg/simulation/sim_02052016/tmp/res_all_adjusted_orderdict_model1.pkl", 'rb') as f:
    res_all_adjusted_orderlist_unpickled = pickle.load(f)

adjusted_r2 = OrderedDict(res_all_adjusted_orderlist_unpickled).values()
plt.scatter(np.arange(50), adjusted_r2)

res_all_adjusted_orderlist_unpickled[38]
res_all_adjusted_orderlist_unpickled[39]
