__author__ = 'panc'

'''
Analyze the .pkl files in "validate_exhaustive_model*_lsra0" folders.

The training method used to produced *_lsra0.pkl files use kernel
regression with intercepts. Thus we append the file name with "a0".
'''

import sys, os
import pickle
import glob
from collections import defaultdict, OrderedDict
import numpy as np
import operator
import itertools
import matplotlib.pyplot as plt

from okgtreg import *


class ValidateExhaustiveLsrAnalyzer(object):
    def __init__(self, pkl_file_path=None, pkl_file=None, true_gstruct=None):
        if pkl_file_path is not None:
            with open(pkl_file_path, 'rb') as f:
                self.res = pickle.load(f)
        elif pkl_file is not None:
            self.res = pickle.load(pkl_file)
        else:
            raise ValueError("** [ERROR] please provide either a pkl file path or pkl file object! **")

        self.true_gstruct = true_gstruct

    def bestParameters(self):
        # For each simulation, what are the tuning parameters
        # corresponding to the lowest test error
        return self.res['bestTuningParameters']

    def bestGroupStructure(self):
        return Group(group_struct_string=self.res['bestGroupStructure'])

    def bestTestError(self):
        test_error_dict = self.res['test_error']
        return min(test_error_dict.values())

    def testErrorOfTrueGroupStructure(self):
        test_error_dict = self.res['test_error']
        return test_error_dict.get(self.true_gstruct.__str__(), -1)


class ValidateExhaustiveLsrAnalyzerAll(object):
    def __init__(self, pkl_folder, true_gstruct=None):
        self.res_list = []
        counter = 0
        for pkl_file in glob.glob(pkl_folder + '/' + "*.pkl"):
            counter += 1
            print '[%d] %s' % (counter, pkl_file)
            self.res_list.append(ValidateExhaustiveLsrAnalyzer(pkl_file_path=pkl_file,
                                                               true_gstruct=true_gstruct))
        self.true_gstruct = true_gstruct

    def bestGroupStructures(self):
        gstruct_freq_dict = {}
        for res in self.res_list:
            gstruct = res.bestGroupStructure()
            gstruct_freq_dict[gstruct.__str__()] = gstruct_freq_dict.get(gstruct.__str__(), 0) + 1
        return gstruct_freq_dict
        # return [res.bestGroupStructure() for res in self.res_list]

    def bestParameters(self):
        params_freq_dict = {}
        for res in self.res_list:
            params_tuple_list = res.bestParameters()
            for params_tuple in params_tuple_list:
                params_freq_dict[params_tuple] = params_freq_dict.get(params_tuple, 0) + 1
        return params_freq_dict

    def freqTrueGroupStructureBeingTheBest(self):
        return self.bestGroupStructures().get(self.true_gstruct.__str__(), 0)

    def bestTestErrors(self):
        return [res.bestTestError() for res in self.res_list]

    def testErrorsOfTrueGroupStructure(self):
        return [res.testErrorOfTrueGroupStructure() for res in self.res_list]

    def plotErrors(self):
        plt.plot(range(1, len(self.res_list) + 1), self.bestTestErrors())
        plt.plot(range(1, len(self.res_list) + 1), self.testErrorsOfTrueGroupStructure(), color='red')
        plt.legend(['best test error', 'test error for true group structure'], loc="upper right")
        return True

    def selectionFreqOfTrueGroupStructureForEachParameter(self, printing=False):
        # count selection freq of group structure after TRAINING
        #   by using each (mu, alpha) pair
        select_gentor = (Res.res['select'] for Res in self.res_list)
        select_all_dict = defaultdict(dict)
        for select_res in select_gentor:
            for k, v in select_res.iteritems():
                select_all_dict[k][v] = select_all_dict[k].get(v, 0) + 1  # Fancy! :)
        # frequency of the true group structure
        select_true_dict = {}
        for k, v in select_all_dict.iteritems():
            select_true_dict[k] = v.get(self.true_gstruct.__str__(), 0)

        if printing:
            select_true_dict2 = defaultdict(dict)
            for k, v in select_true_dict.iteritems():
                select_true_dict2[k[0]][k[1]] = v
            # Reference: sort a dictionary by key
            #   http://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key
            select_true_dict2_order1 = {}
            for k, v in select_true_dict2.iteritems():
                select_true_dict2_order1[k] = OrderedDict(sorted(v.items()))
            select_true_dict2_order2 = OrderedDict(sorted(select_true_dict2_order1.items()))

            print "{0:10}   {1:10}   {2}".format("mu", "alpha", "consistent.selection.freq")
            for k, v in select_true_dict2_order2.iteritems():
                for k2, v2 in v.iteritems():
                    # Reference: print numbers in scientific notation form
                    #   https://mail.python.org/pipermail/tutor/2008-June/062649.html
                    print "{0:10} : {1:10} : {2}".format("%.5e" % (k), "%.02f" % (k2), v2)

        return select_true_dict  # {(mu, alpha) : freq}

    def selectFreqOfTrueGroupStructureAfterValidation(self, printing=False):
        # For each model (including 100 simulations), what is the frequency
        #   that the true group structure is selected after VALIDATION.
        # At the same, an user has the option to print all selected group
        #   structures and their selection frequencies.
        select_gentor = (Res.res['bestGroupStructure'] for Res in self.res_list)
        select_freq_dict = {}
        for gstruct_str in select_gentor:
            select_freq_dict[gstruct_str] = select_freq_dict.get(gstruct_str, 0) + 1
        if printing:
            sortedGstruct_str_list = sorted(select_freq_dict, key=select_freq_dict.get,
                                            reverse=True)
            for gstruct_str in sortedGstruct_str_list:
                print "{0:10} : {1}".format(gstruct_str, select_freq_dict[gstruct_str])

        return select_freq_dict[self.true_gstruct.__str__()]


class ValidateExhaustiveLsrAnalyzerAllModels(object):
    pass

####################
# Start to analyze #
####################
if __name__ == '__main__':
    true_gstructs_dict = {3: Group([1], [2], [3], [4], [5], [6]),
                          2: Group([1], [2, 3], [4, 5, 6]),
                          5: Group([1, 3], [2], [4, 5, 6]),
                          12: Group([1, 2], [3, 4], [5, 6]),
                          7: Group([1, 2, 3, 4], [5, 6]),
                          9: Group([1, 2, 3, 4, 5, 6])}

    ########################################
    # Analyze all .pkl files in one folder #
    ########################################
    model_id = 12  # 2, 3, 5, 7, 9, 12
    pkl_folder = "okgtreg/simulation/sim_02052016/validate_exhaustive_model" + str(model_id) + "_lsra0"

    FolderAnalyzer = ValidateExhaustiveLsrAnalyzerAll(pkl_folder, true_gstructs_dict[model_id])
    FolderAnalyzer.freqTrueGroupStructureBeingTheBest()
    FolderAnalyzer.selectionFreqOfTrueGroupStructureForEachParameter()
    FolderAnalyzer.bestParameters()
    FolderAnalyzer.bestGroupStructures()
    FolderAnalyzer.selectFreqOfTrueGroupStructureAfterValidation(printing=True)

    ################################################
    # Selection frequency of true group structures #
    # for ALL FOLDERS/MODELS (after training).     #
    # This is the results for Section 4.1 in NIPS  #
    # 2016 paper.                                  #
    ################################################
    selection_freq_dict = defaultdict(list)
    # model_ids = [3, 2, 5, 12, 9] # for NIPS 2016 paper
    model_ids = [3, 2, 5, 9] # for thesis defense
    # for model_id in [3, 2, 5, 12, 7, 9]: # for UAI 2016 paper
    for model_id in model_ids:
        print "Model: ", model_id
        sim_folder = "okgtreg/simulation/sim_02052016"
        pkl_folder = "validate_exhaustive_model" + str(model_id) + "_lsra0"
        FolderAnalyzer = ValidateExhaustiveLsrAnalyzerAll(sim_folder + '/' + pkl_folder,
                                                          true_gstructs_dict[model_id])
        for k, v in FolderAnalyzer.selectionFreqOfTrueGroupStructureForEachParameter().iteritems():
            selection_freq_dict[k].append(v)

    selection_freq_dict_orderByKey = OrderedDict(sorted(selection_freq_dict.items()))

    # ===
    # Print the latex table of (successful identification) frequencies:
    #   \mu  \alpha  M1  M2  M3  M4  ~M5~  M6
    # M5 is removed for the NIPS 2016 paper.
    # ===
    num_models = len(model_ids)
    for k, v in selection_freq_dict_orderByKey.iteritems():
        print "{0:10} & {1:5} & {2:5} & {3:5} & {4:5} & {5:5} & {6:5} \\\\".format(
            "%.4e" % (k[0]), "%.02f" % (k[1]), v[0], v[1], v[2], v[3], v[4])

        # format_str = '{0:10} & {1:5} & ' + \
        #     ' & '.join(['{'+str(i)+':'+str(5)+'}' for i in np.linspace(1, num_models, num_models).astype(int) + 1])

    # ===
    # The latex table is too large for the UAI 2016 paper,
    #   Instead, for each model, make a line plot where
    #   the ups and downs show the selection frequency
    # ===
    freq_array = np.array(selection_freq_dict_orderByKey.values())
    fig, axarr = plt.subplots(3, 2, sharex=True, sharey=True)
    for i in range(3):
        for j in range(2):
            ## Reference: plot with dot-and-line
            ##  http://matplotlib.org/users/pyplot_tutorial.html#working-\
            ##  with-multiple-figures-and-axes
            axarr[i, j].plot(range(1, 51), freq_array[:, i * 2 + j], 'bo',
                             range(1, 51), freq_array[:, i * 2 + j], 'k')
            axarr[i, j].set_ylim([-10, 110])
            axarr[i, j].set_xlim([1, 50])
            axarr[i, j].set_title("Model " + str(i * 2 + j + 1))
    ## Reference: Reduce left and right margins in matplotlib plot
    ##  http://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    fig.tight_layout()
    fig.show()

    # ===
    # It turns out (based on the UAI 2016 reviews) that the
    #   line plots may cause confusion in understanding. Now,
    #   I am trying to use a 3d heatmap for each model to show
    #   the frequencies over the grid of (\mu, \alpha) pairs.
    # ===
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    ## Set x (equal spaced in log-scale), y axis values
    xgv = np.unique([k[0] for (k,v) in selection_freq_dict_orderByKey.iteritems()])
    xgv = np.log(xgv)
    ygv = np.unique([k[1] for (k,v) in selection_freq_dict_orderByKey.iteritems()])
    ## Create a mesh grid from the x and y values
    [X,Y] = np.meshgrid(xgv, ygv)
    ## Plot, one image for each model.
    ## The values of Z (frequencies) change for each plot.
    fig = plt.figure(figsize=(12, 6))
    for i in range(4):
        ## values for z
        Z = np.array([v[i] for (k,v) in selection_freq_dict_orderByKey.iteritems()]).reshape(X.shape, order='F')
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

    ###############################################
    # Selection frequency of true group structure #
    # after training + validation.                #
    # Results for Section 4.2 in NIPS 2016 paper. #
    ###############################################
    select_freq_dict = {}
    select_freq_params_dictList = []
    # model_ids = [3, 2, 5, 12, 9] # for NIPS 2016
    model_ids = [3, 2, 5, 9] # for defense
    # for model_id in [3, 2, 5, 12, 7, 9]:
    for model_id in model_ids:
        print "Model: ", model_id
        # create a Folder Analyzer for model_id
        sim_folder = "okgtreg/simulation/sim_02052016"
        pkl_folder = "validate_exhaustive_model" + str(model_id) + "_lsra0"
        FolderAnalyzer = ValidateExhaustiveLsrAnalyzerAll(sim_folder + '/' + pkl_folder,
                                                          true_gstructs_dict[model_id])
        # frequency that the true group structure is the best
        true_gstruct_str = true_gstructs_dict[model_id].__str__()
        select_freq_dict[true_gstruct_str] = \
            FolderAnalyzer.selectFreqOfTrueGroupStructureAfterValidation()
        # frequency of (mu, alpha) selecting the true group structure as the best
        mu_size = 5
        alpha_size = 10
        muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
        alphaList = np.arange(1, alpha_size + 1)
        ## initialize dict
        select_freq_param_dict = {}
        for mu, alpha in itertools.product(muList, alphaList):
            select_freq_param_dict[(mu, alpha)] = 0
        for Analyzer in FolderAnalyzer.res_list:
            if Analyzer.bestGroupStructure().__str__() == true_gstruct_str:
                select_dict = Analyzer.res['select']
                for k, v in select_dict.iteritems():
                    if v == true_gstruct_str:
                        select_freq_param_dict[k] = select_freq_param_dict.get(k, 0) + 1

        select_freq_params_dictList.append(select_freq_param_dict)

    # Save the objects in case Python crashes...
    with open("okgtreg/simulation/sim_02052016/tmp/select_freq_dict.pkl", 'wb') as f:
        pickle.dump(select_freq_dict, f)

    with open("okgtreg/simulation/sim_02052016/tmp/select_freq_params_dictList.pkl", 'wb') as f:
        pickle.dump(select_freq_params_dictList, f)

    # Load pkl
    with open("okgtreg/simulation/sim_02052016/tmp/select_freq_dict.pkl", 'rb') as f:
        select_freq_dict = pickle.load(f)

    with open("okgtreg/simulation/sim_02052016/tmp/select_freq_params_dictList.pkl", 'rb') as f:
        select_freq_params_dictList = pickle.load(f)

    # ===
    # Format as a Latex table
    # ===
    print "{0:2} & {1:2} & {2:2} & {3:2} & {4:2} \\\\".format('M1', 'M2', 'M3', 'M4', 'M5')
    # print "{0:2} & {1:2} & {2:2} & {3:2} & {4:2} & {5:2} \\\\".format(*tuple(select_freq_dict.values()))
    for model_id in model_ids:
        print "{0:2} & ".format(select_freq_dict[true_gstructs_dict[model_id].__str__()]),
    print "{0:2} \\\\".format(select_freq_dict[true_gstructs_dict[model_id].__str__()])

    # Find optimal parameters
    select_freq_params_dictList[0]
    select_freq_params_dictList[1]
    select_freq_params_dictList[2]
    select_freq_params_dictList[3]
    select_freq_params_dictList[4]

    # ===
    # LINE PLOTS.
    # For each model, plot the frequency curve to show
    #   how often (mu, alpha) being optimal after validation
    # ===
    fig, axarr = plt.subplots(3, 2, sharex=True, sharey=True)
    for i in range(3):
        for j in range(2):
            ## Need to preserve the order of the frequencies
            ##   according to the increasing order of (mu, alpha)
            freq_to_plot = [v for k, v in sorted(select_freq_params_dictList[i * 2 + j].items())]
            ## Reference: plot with dot-and-line
            ##  http://matplotlib.org/users/pyplot_tutorial.html#working-with-multiple-figures-and-axes
            axarr[i, j].plot(range(1, 51), freq_to_plot, 'bo',
                             range(1, 51), freq_to_plot, 'k')
            axarr[i, j].set_ylim([-10, 110])
            axarr[i, j].set_xlim([0, 51])
            axarr[i, j].set_title("Model " + str(i * 2 + j + 1))
    ## Reference: Reduce left and right margins in matplotlib plot
    ##  http://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    fig.tight_layout()
    fig.show()

    # ===
    # 3D SURFACE PLOTS.
    # It turns out (based on the UAI 2016 reviews) that the
    #   line plots may cause confusion in understanding. Now,
    #   I am trying to use a 3D heatmap for each model to show
    #   the frequencies over the grid of (\mu, \alpha) pairs.
    # ===
    ## Set x (equal spaced in log-scale), y axis values
    xgv = np.unique([k[0] for (k, v) in select_freq_params_dictList[0].iteritems()])
    xgv = np.log(xgv)
    ygv = np.unique([k[1] for (k, v) in select_freq_params_dictList[0].iteritems()])
    ## Create a mesh grid from the x and y values
    [X, Y] = np.meshgrid(xgv, ygv)
    ## Plot, one image for each model.
    ## The values of Z (frequencies) change for each plot.
    fig = plt.figure(figsize=(12, 6))
    for i in range(4):
        ## values for z
        Z = np.array([v for (k, v) in sorted(select_freq_params_dictList[i].items())]).reshape(X.shape, order='F')
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.plot_surface(X, Y, Z,
                        rstride=1, cstride=1,
                        # cmap=plt.cm.jet, #plt.cm.CMRmap, #plt.cm.Spectral,
                        linewidth=0.5,
                        # antialiased=True,
                        alpha=0.3)
        ax.set_title("Model " + str(i + 1))
        cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z) - 20, cmap=cm.coolwarm, alpha=0.7)
        cset = ax.contourf(X, Y, Z, zdir='x', offset=np.min(X) - 5, cmap=cm.coolwarm, alpha=0.7)
        cset = ax.contourf(X, Y, Z, zdir='y', offset=np.max(Y) + 5, cmap=cm.coolwarm, alpha=0.7)
        ax.set_xlabel(r'$\log(\mu)$')
        ax.set_xlim(np.min(X) - 5, np.max(X))
        ax.set_ylabel(r'$\alpha$')
        ax.set_ylim(np.min(Y), np.max(Y) + 5)
        # ax.set_zlabel('Z')
        ax.set_zlim(np.min(Z) - 20, 100)
        # ax.view_init(elev=30, azim=35)
    fig.tight_layout(pad=1.5, w_pad=1.5, h_pad=2.0)
    fig.show()