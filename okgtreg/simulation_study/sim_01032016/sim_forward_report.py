__author__ = 'panc'

import pickle
from okgtreg.simulation.sim_01032016.utility import *


pklfile = open('okgtreg/simulation/sim_01032016/sim_forward.py.pkl', 'rb')
groups, r2s = pickle.load(pklfile)
pklfile.close()

printGroupingFrequency(groups)
printGroupFrequency(groups)

np.mean(r2s)