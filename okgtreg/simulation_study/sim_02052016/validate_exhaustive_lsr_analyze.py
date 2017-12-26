__author__ = 'panc'

'''
Analyze the .pkl files in "validate_exhaustive_model*_lsr" folders.
'''

import sys, os
import pickle
import glob
from collections import defaultdict, OrderedDict
import numpy as np
import operator
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


if __name__ == '__main__':
    model_id = 4
    pkl_folder = "okgtreg/simulation/sim_02052016/validate_exhaustive_model" + str(model_id) + "_lsra0"
    true_gstructs_dict = {1: Group([1, 2], [3, 4], [5, 6]),
                          2: Group([1], [2, 3], [4, 5, 6]),
                          3: Group([1], [2], [3], [4], [5], [6]),
                          4: Group([1, 2, 3, 4, 5, 6]),
                          5: Group([1, 3], [2], [4, 5, 6]),
                          6: Group([1, 2, 3, 4], [5, 6])}

    ########################
    # Analyze one pkl file #
    ########################
    pkl_file = "validate_exhaustive-model4-seed1-201602231911.pkl"
    pkl_file_path = pkl_folder + '/' + pkl_file
    Analyzer1 = ValidateExhaustiveLsrAnalyzer(pkl_file_path)
    # Best Group Structure
    Analyzer1.bestGroupStructure()
    # Best Parameters
    Analyzer1.bestParameters()
    # Best/Smallest test error
    Analyzer1.bestTestError()
    # Test error for the true group structure
    Analyzer1.testErrorOfTrueGroupStructure()

    #####################################
    # Analyze all pkl files in a folder #
    #####################################
    FolderAnalyzer = ValidateExhaustiveLsrAnalyzerAll(pkl_folder, true_gstructs_dict[model_id])
    # Frequency of the parameter pairs to be the best
    FolderAnalyzer.bestParameters()
    # Frequency of the group structures to be the best
    FolderAnalyzer.bestGroupStructures()
    # Frequency that the true group structure is selected to be the best
    FolderAnalyzer.freqTrueGroupStructureBeingTheBest()
    # Plot the best test errors and the test errors for the true group
    # structure for each simulation
    best_errors = FolderAnalyzer.bestTestErrors()
    true_gstruct_errors = FolderAnalyzer.testErrorsOfTrueGroupStructure()
    FolderAnalyzer.plotErrors()

    best_errors_max_idx = best_errors.index(max(best_errors))
    print best_errors_max_idx, ':', max(best_errors)
    Analyzer_max = FolderAnalyzer.res_list[best_errors_max_idx]

    # Reproducing the data for Model 4 Seed 48
    from okgtreg.simulation.sim_02052016.model import selectModel

    ## Reproduce the data
    model = selectModel(model_id)
    seed_num = best_errors_max_idx + 1  # 34
    ntrain = 500  # train size
    ntest = 500  # test size
    np.random.seed(seed_num)
    train_data, true_train_group, train_g = model(ntrain)
    test_data, true_test_group, test_g = model(ntest)
    ## Predicted test value from the simulation
    kernel = Kernel('gaussian', sigma=0.5)


    def predict(test_data, kernel, gstruct, fns_est):
        parameters = Parameters(gstruct, kernel, [kernel] * gstruct.size)
        parameterizedTestData = ParameterizedData(test_data, parameters)
        test_ghat_list = [fns_est[i](parameterizedTestData.getXFromGroup(i))
                          for i in range(1, gstruct.size + 1)]
        # test_ghat = sum(test_ghat_list) + fns_est[0] # with intercept
        test_ghat = sum(test_ghat_list)  # without intercept
        return test_ghat


    ### Under true group structure
    true_gstruct_m4 = true_gstructs_dict[model_id]
    print "true group structure:", true_gstruct_m4
    true_fns_est = Analyzer_max.res['train_f'][true_gstruct_m4.__str__()]
    true_test_ghat = predict(test_data, kernel, true_gstruct_m4, true_fns_est)
    ### Under the selected best group structure
    best_gstruct_m4s34 = Analyzer_max.bestGroupStructure()
    print "selected group structure:", best_gstruct_m4s34
    best_fns_est = Analyzer_max.res['train_f'][best_gstruct_m4s34.__str__()]
    best_test_ghat = predict(test_data, kernel, best_gstruct_m4s34, best_fns_est)

    ### plot histograms
    plt.hist(train_g, 50, color='grey')  # train data response

    plt.hist(test_g, 50, color='blue')  # test data response
    plt.hist(best_test_ghat, 50, color='green',
             edgecolor="none")  # test response predicted under the best group structure
    plt.hist(true_test_ghat, 50, color='red',
             edgecolor="none")  # test response predicted under the true group structure

    ### plot the response g as curves
    plt.plot(range(1, 501), test_g, color='black')
    plt.plot(range(1, 501), true_test_ghat, color='blue')
    plt.plot(range(1, 501), best_test_ghat, color='green')

    ## Fitted train value from the simulation
    true_train_ghat = predict(train_data, kernel, true_gstruct_m4, true_fns_est)
    plt.hist(true_train_ghat, 50)

    ## goodness of train fit
    1 - np.sum((train_g - true_train_ghat) ** 2) / np.sum((train_g - np.mean(train_g)) ** 2)

    ## generalization error
    print np.sum((test_g - true_test_ghat) ** 2) / ntest
    print np.sum((test_g - best_test_ghat) ** 2) / ntest

    Analyzer_max.bestTestError()

    # ////////////////////////////////////////////////
    # Print correct selection freq
    bestParameters_dic = FolderAnalyzer.bestParameters()
    correctSelectFreq = FolderAnalyzer.freqTrueGroupStructureBeingTheBest()
    trueParameters_dict = defaultdict(list)  # parameters that select true group structure as the best
    for k, v in bestParameters_dic.iteritems():
        if v == correctSelectFreq:
            trueParameters_dict[k[0]].append(k[1])
    ## print mu and alpha's for each mu
    for k, v_list in trueParameters_dict.items():
        # print type(k), ":", type(v_list)
        print "{0:10} & ".format("%.04e" % k),
        for i in range(len(v_list)):
            if i != len(v_list) - 1:
                print ("{0:3} & ").format("%d" % v_list[i]),
            else:
                print ("{0:3}").format("%d" % v_list[i])
