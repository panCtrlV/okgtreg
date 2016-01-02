import pickle

pklfile = open("okgtreg/simulation/sim_01022016/sim.pkl", 'rb')
groups, r2s = pickle.load(pklfile)
pklfile.close()

