__author__ = 'panc'

'''
Analyze validate_exhaustive_bootstrap
'''

import pickle
import glob
from collections import defaultdict
import numpy as np

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

res_all_dict.items(0)
# Calculate mean r2 and std dev for each (mu, alpha)
{k: (np.mean(v), np.std(v)) for k, v in res_all_dict.items()}

{k: max(v) for k, v in res_all_dict.items()}
