__author__ = 'panc'

import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
import operator
import os

from okgtreg.Group import Group


# Analyze one pkl file
class ValidateBackwardAnalyzer(object):
    def __init__(self, f=None, pkl_file_path=None, true_gstruct=None):
        if f is not None:
            self.res = pickle.load(f)
        elif pkl_file_path is not None:
            with open(pkl_file_path, 'rb') as f:
                self.res = pickle.load(f)

        self.true_gstruct = true_gstruct

    def selectedGroupStructureFromTrain(self):
        return self.res['select']


# Analyze a folder
class ValidateBackwardFolderAnalyzer(object):
    def __init__(self, pkl_folder_path, true_gstruct=None, print_path=False):
        '''
        Reference for listing all files with certain extension
        in a folder: https://docs.python.org/2/library/glob.html
        '''
        self.Analyzer_list = []
        # TODO: in what order the files are read
        for i, fpath in enumerate(glob.glob(pkl_folder_path + '/' + "*.pkl")):
            if print_path:
                print '[%d]' % (i + 1), fpath
            self.Analyzer_list.append(ValidateBackwardAnalyzer(pkl_file_path=fpath,
                                                               true_gstruct=true_gstruct))
        self.true_gstruct = true_gstruct

    def selectTrueFromTrain(self):
        mu_size = 5
        alpha_size = 10
        muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
        alphaList = np.arange(1, alpha_size + 1)
        # initialize dict
        param_trueFeq_dict = {}
        for mu, alpha in itertools.product(muList, alphaList):
            param_trueFeq_dict[(mu, alpha)] = 0
        for Analyzer in self.Analyzer_list:
            for k, v in Analyzer.selectedGroupStructureFromTrain().iteritems():
                if v == self.true_gstruct.__str__():
                    param_trueFeq_dict[k] = param_trueFeq_dict.get(k, 0) + 1
        return param_trueFeq_dict


if __name__ == '__main__':
    module_folder = '/Users/panc25/Dropbox/Research/Publications/paper_OKGT/software/okgtreg'

    true_group_structures = {3: Group([1], [2], [3], [4], [5], [6]),
                             2: Group([1], [2, 3], [4, 5, 6]),
                             5: Group([1, 3], [2], [4, 5, 6]),
                             12: Group([1, 2], [3, 4], [5, 6]),
                             7: Group([1, 2, 3, 4], [5, 6]),
                             9: Group([1, 2, 3, 4, 5, 6])}

    model_id = 5
    sim_folder = "okgtreg/simulation/sim_02052016"
    pkl_folder = "validate_backward_model" + str(model_id) + "_lra0"
    sample_pkl_file = "validate_backward_lra0-model5-seed1-20160228-100535-219518.pkl"
    Analyzer = ValidateBackwardAnalyzer(
        pkl_file_path=os.path.join(module_folder, sim_folder, pkl_folder, sample_pkl_file),
        true_gstruct=true_group_structures[model_id])

    # Selected group structure from training for each (mu, alpha)
    Analyzer.selectedGroupStructureFromTrain()

    ######################
    # Analyze one folder #
    ######################
    # for model_id in [3,2,5,12,7,9]:
    model_id = 5
    sim_folder = "okgtreg/simulation/sim_02052016"
    pkl_folder = "validate_backward_model" + str(model_id) + "_lra0"
    FolderAnalyzer = ValidateBackwardFolderAnalyzer(
        pkl_folder_path=sim_folder + '/' + pkl_folder,
        true_gstruct=true_group_structures[model_id])

    FolderAnalyzer.selectTrueFromTrain()

    #######################
    # Analyze ALL folders #
    #######################

    # ===
    #  For each model, plot the frequency curve to show
    #    how often (mu, alpha) being optimal after validation
    # ===
    #
    # Import all simulation results as FolderAnalyzer,
    # which are saved in a list
    sim_folder = "okgtreg/simulation/sim_02052016"
    FolderAnalyzer_list = []
    model_ids = [3, 2, 5, 12, 9] # for NIPS 2016 and ICML 2017
    # model_ids = [3, 2, 5, 9] # for defense
    # for model_id in [3, 2, 5, 12, 7, 9]:
    for model_id in model_ids:
        print "Model: %d" % model_id
        print "Group structure: %s" % true_group_structures[model_id].__str__()
        pkl_folder = "validate_backward_model" + str(model_id) + "_lra0"
        FolderAnalyzer_list.append(
            ValidateBackwardFolderAnalyzer(
                pkl_folder_path=os.path.join(module_folder, sim_folder, pkl_folder),
                true_gstruct=true_group_structures[model_id])
        )

    # Selection frequency of true group structures for each (mu, alpha) pair
    FolderAnalyzer_list[0].selectTrueFromTrain()
    FolderAnalyzer_list[1].selectTrueFromTrain()
    FolderAnalyzer_list[2].selectTrueFromTrain()
    FolderAnalyzer_list[3].selectTrueFromTrain()
    FolderAnalyzer_list[4].selectTrueFromTrain()

    # Select one of the max frequency for each model
    print '\\hline'
    print "Model & Max freq. & $\\mu$ & $\\alpha$ \\\\"
    print '\\hline'
    fmtStr = "{:s} & {:5d} & {:13.04e} & {:5d} \\\\"
    for i in range(5):
        freqDict = FolderAnalyzer_list[i].selectTrueFromTrain()
        maxItem = max(freqDict.iteritems(), key=operator.itemgetter(1))
        print fmtStr.format('M'+str(i+1), maxItem[1], maxItem[0][0], maxItem[0][1])
    print '\\hline'


    # Plot
    fig, axarr = plt.subplots(3, 2, sharex=True, sharey=True)
    for i in range(3):
        for j in range(2):
            select_freq_params_dict = FolderAnalyzer_list[i * 2 + j].selectTrueFromTrain().items()

            ## Need to preserve the order of the frequencies
            ##   according to the increasing order of (mu, alpha)
            freq_to_plot = [v for k, v in sorted(select_freq_params_dict)]
            ## Reference: plot with dot-and-line
            ##  http://matplotlib.org/users/pyplot_tutorial.html#working-with-multiple-figures-and-axes
            # axarr[i,j].plot(range(1, 51), freq_to_plot, 'bo',
            #                 range(1, 51), freq_to_plot, 'k')
            ## Using the following two lines for plotting
            ##  has better control on the dot size
            axarr[i, j].scatter(range(1, 51), freq_to_plot, s=10, linewidth=0)
            axarr[i, j].plot(range(1, 51), freq_to_plot, 'k')

            axarr[i, j].set_ylim([-10, 110])
            axarr[i, j].set_xlim([0, 51])
            axarr[i, j].set_title("Model " + str(i * 2 + j + 1))
    ## Reference: Reduce left and right margins in matplotlib plot
    ##  http://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    fig.tight_layout()
    fig.show()

    # ===
    # It turns out (based on the UAI 2016 reviews) that the
    #   line plots may cause confusion in understanding. Now,
    #   I am trying to use a 3D heatmap for each model to show
    #   the frequencies over the grid of (\mu, \alpha) pairs.
    # ===
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    ## Set x (equal spaced in log-scale), y axis values
    xgv = np.unique([k[0] for (k,v) in sorted(FolderAnalyzer_list[0].selectTrueFromTrain().items())])
    xgv = np.log(xgv)
    ygv = np.unique([k[1] for (k,v) in sorted(FolderAnalyzer_list[0].selectTrueFromTrain().items())])
    ## Create a mesh grid from the x and y values
    [X,Y] = np.meshgrid(xgv, ygv)
    ## Plot, one image for each model.
    ## The values of Z (frequencies) change for each plot.
    fig = plt.figure(figsize=(12, 6))
    for i in range(4):
        ## values for z
        Z = np.array([v for (k,v) in sorted(FolderAnalyzer_list[i].selectTrueFromTrain().items())]).reshape(X.shape, order='F')
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot_surface(X, Y, Z,
                        rstride=1, cstride=1,
                        #cmap=plt.cm.jet, #plt.cm.CMRmap, #plt.cm.Spectral,
                        linewidth=0.5,
                        # antialiased=True,
                        alpha=0.3)
        ax.set_title("Model " + str(i+1))
        cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z)-20, cmap=cm.coolwarm, alpha=0.7)
        cset = ax.contourf(X, Y, Z, zdir='x', offset=np.min(X)-5, cmap=cm.coolwarm, alpha=0.7)
        cset = ax.contourf(X, Y, Z, zdir='y', offset=np.max(Y)+5, cmap=cm.coolwarm, alpha=0.7)
        ax.set_xlabel(r'$\log(\mu)$')
        ax.set_xlim(np.min(X)-5, np.max(X))
        ax.set_ylabel(r'$\alpha$')
        ax.set_ylim(np.min(Y), np.max(Y)+5)
        # ax.set_zlabel('Z')
        ax.set_zlim(np.min(Z)-20, 100)
        # ax.view_init(elev=30, azim=35)
    fig.tight_layout(pad=1.5, w_pad=1.5, h_pad=2.0)
    fig.show()
