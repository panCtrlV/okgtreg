__author__ = 'panc'

import pickle
from okgtreg.simulation.sim_01032016_2.utility import *

pklfile = open('okgtreg/simulation/sim_01032016_2/sim_forward.py.pkl', 'rb')
groups, r2s = pickle.load(pklfile)
pklfile.close()

np.mean(r2s)

printGroupingFrequency(groups)
printGroupFrequency(groups)
