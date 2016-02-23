import sys, os
import pickle
import glob
from collections import defaultdict, OrderedDict
import numpy as np
import operator
import matplotlib.pyplot as plt

from okgtreg import *


# Processor for a single .pkl file
class ValidateExhaustiveProcessor(object):
    def __init__(self, f=None, pkl_file_path=None, true_group_struct=None):
        if f is not None:
            self.res = pickle.load(f)
        elif pkl_file_path is not None:
            with open(pkl_file_path, 'rb') as f:
                self.res = pickle.load(f)

        self.true_group_struct = true_group_struct

    # best group strcuture determined by the
    # one-fold validation for a given data set.
    def bestGroupStructure(self):
        return self.res['bestGroupStructure']

    # best group strcutures after imposing penalty,
    # one for each (mu, alpha) pair.
    #
    # Reference for spacing and aligning strings in Python:
    #   http://stackoverflow.com/questions/10623727/python-spacing-and-aligning-strings
    def bestGroupStructuresFromTrainWithPenalty(self, printing=False):
        if printing:
            for k, v in self.res['select'].iteritems():
                if self.true_group_struct is not None:
                    # If the true group structure is provided,
                    # the (mu, alpha) pair that successfully selected
                    # the true group structure will be indicated by a "*"
                    if v == self.true_group_struct.__str__():
                        print "{0:30} : {1} *".format(k, v)
                    else:
                        print "{0:30} : {1}".format(k, v)
                else:
                    print "{0:30} : {1}".format(k, v)
        return self.res['select']

    # show the test R2 for each selected group structure
    # from the training phase
    # Note: in the one-fold validation, a group structure
    #       can be selected by multiple (mu, alpha) pairs
    #       as optimal from the training phase. So the 
    #       number of group structures in the test dictionary
    #       is usually much smaller than that in the train
    #       dictionary.
    def testResults(self, printing=False, sort=True):
        test_dict = self.res['test']
        if printing:
            if sort:
                # list of sorted group structures
                group_struct_list = sorted(test_dict, key=test_dict.get, reverse=True)
            else:
                group_struct_list = test_dict.keys()

            for g in group_struct_list:
                print "{0:30} : {1}".format(g, test_dict[g])

        return test_dict

    def printAmiableRankingInTrainingWithoutPenalty(self):
        res_train = self.res['train']
        sorted_group_struct = sorted(res_train, key=res_train.get, reverse=True)
        amiables = self.true_group_struct.amiableGroupStructures()
        amiables_str = [gstruct.__str__() for gstruct in amiables]
        # An amiable group structure is indicated by "*",
        # the true group structure is indicated by "true"
        for item in sorted_group_struct:
            if item in amiables_str:
                print item, " *"
            elif item == self.true_group_struct.__str__():
                print item, " true"
            else:
                print item

        return True

    def rankOfAmiableGroupStructuresInTrainingWithoutPenalty(self):
        res_train = self.res['train']
        sorted_gstruct_str = [gstruct.__str__() for gstruct in
                              sorted(res_train, key=res_train.get, reverse=True)]
        amiables = self.true_group_struct.amiableGroupStructures()
        amiables_str = [gstruct.__str__() for gstruct in amiables]
        rank_dict = {}
        # Reference: find index of an element in a list
        # http://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python
        for gstruct_str in amiables_str:
            rank_dict[gstruct_str] = sorted_gstruct_str.index(gstruct_str) + 1
        return rank_dict

    def testR2ForAllParameters(self, ordered=True, plotting=True):
        res_select = self.res['select']
        res_test = self.res['test']

        testR2ForAllParameters_dict = {k: res_test[v] for k, v in res_select.items()}
        if ordered:
            testR2ForAllParameters_orderdict = OrderedDict(
                sorted(testR2ForAllParameters_dict.items(), key=operator.itemgetter(1), reverse=True)
            )
            if plotting:
                plt.scatter(np.arange(50), testR2ForAllParameters_orderdict.values())
            return testR2ForAllParameters_orderdict
        else:
            return testR2ForAllParameters_dict


# Processor for all .pkl for a given model
class ValidateExhaustiveProcessorAll(object):
    def __init__(self, pkl_folder_path, true_group_struct=None):
        '''
        Reference for listing all files with certain extension
        in a folder: https://docs.python.org/2/library/glob.html
        '''
        self.res_list = []
        # TODO: in what order the files are read
        for fpath in glob.glob(pkl_folder_path + '/' + "*.pkl"):
            self.res_list.append(ValidateExhaustiveProcessor(pkl_file_path=fpath,
                                                             true_group_struct=true_group_struct))

        self.true_group_struct = true_group_struct

    # each ,pkl file has a best group structure
    # we list all of them
    def bestGroupStructures(self):
        return [res.bestGroupStructure() for res in self.res_list]

    ## For a given model, compile the rank of amiable group structures
    ## for each simulation. The table looks like:
    ##
    ##              sim1    sim2    sim3 ...
    ##  gstruct 1
    ##  gstruct 2
    ##  gstruct 4
    ##  ...
    ##
    def rankOfAmiableGroupStructuresInTrainingWithoutPenalty(self, printing=False):
        # train_dict_gen = (Res.res['train'] for Res in self.res_list)
        # all amiable group structures
        amiables = self.true_group_struct.amiableGroupStructures()
        amiables_str = [gstruct.__str__() for gstruct in amiables]
        all_rank_dict = defaultdict(list)
        for Res in self.res_list:
            rank_res = Res.rankOfAmiableGroupStructuresInTrainingWithoutPenalty()
            for gstruct_str in amiables_str:
                all_rank_dict[gstruct_str].append(rank_res[gstruct_str])

        # print the mean rank and standard deviation for each amiable group structure
        # sort by mean rank
        if printing:
            amiable_mean_rank_dict = {}
            for gstruct in amiables:
                amiable_mean_rank_dict[gstruct.__str__()] = np.mean(all_rank_dict[gstruct.__str__()])
            sorted_amiable_by_mean = sorted(amiable_mean_rank_dict, key=amiable_mean_rank_dict.get)
            print "{0:30}   {1:10}   {2}".format("amiable group structure", "mean rank", "std.dev")
            for gstruct_str in sorted_amiable_by_mean:
                avg_rank = amiable_mean_rank_dict[gstruct_str]
                stdev_rank = np.std(all_rank_dict[gstruct_str])
                print "{0:30} : {1:10} : {2}".format(gstruct_str, avg_rank, stdev_rank)

        return all_rank_dict

    # Percentage of times for each (mu, alpha) that the true group structure
    # is selected during training phase.
    def selectionFreqOfTrueGroupStructureForEachParameter(self, printing=False):
        mu_size = 5
        alpha_size = 10
        muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
        alphaList = np.arange(1, alpha_size + 1)

        # count frequency of each selected group structure
        # for each (mu, alpha) pair
        select_gen = (Res.res['select'] for Res in self.res_list)
        select_all_dict = defaultdict(dict)
        for select_res in select_gen:
            for k, v in select_res.iteritems():
                select_all_dict[k][v] = select_all_dict[k].get(v, 0) + 1  # Fancy! :)

        # frequency of the true group structure
        select_true_dict = {}
        for k, v in select_all_dict.iteritems():
            select_true_dict[k] = v.get(self.true_group_struct.__str__(), 0)

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


if __name__ == "__main__":
    ########################
    # Unpickle a .pkl file #
    ########################

    model_id = 1
    # sim_folder = "/home/panc/research/OKGT/software/okgtreg/okgtreg/simulation/sim_02052016"
    sim_folder = "okgtreg/simulation/sim_02052016"
    pkl_folder = "validate_exhaustive_model" + str(model_id)
    file_name = "validate_exhaustive-model1-seed100-201602190127.pkl"
    true_group_struct = Group([1, 2], [3, 4], [5, 6])  # model 1
    # true_group_struct = Group([1],[2,3],[4,5,6]) # model 2
    # true_group_struct = Group([1],[2],[3],[4],[5],[6]) # model 3
    # true_group_struct = Group([1,2,3,4,5,6]) # model 4
    # true_group_struct = Group([1, 3], [2], [4, 5, 6]) # model 5
    # true_group_struct = Group([1, 2, 3, 4], [5, 6]) # model 6
    amiables = true_group_struct.amiableGroupStructures()

    # construct the object for a single pkl processor
    Res = ValidateExhaustiveProcessor(pkl_file_path=sim_folder + '/' + pkl_folder + '/' + file_name,
                                      true_group_struct=true_group_struct)
    ## best group structure determined by one-folder validation
    ## using this data set
    Res.bestGroupStructure()
    ## the best group structures selected by the training phase
    ## for different (mu, alpha) pairs
    res_train_with_penalty = Res.bestGroupStructuresFromTrainWithPenalty(printing=True)
    ## Check if amiable group structures are top ranked in the
    ## training phase without penalty.
    Res.printAmiableRankingInTrainingWithoutPenalty()
    ## and list the ranks for each amiable group structure
    Res.rankOfAmiableGroupStructuresInTrainingWithoutPenalty()
    ## Plot test R2 for each (mu, alpha) in decreasing order
    Res.testR2ForAllParameters()

    #######################################
    # Unpickle all ,pkl files in a folder #
    #######################################

    # construct the object for all pkl files
    ResAll = ValidateExhaustiveProcessorAll(sim_folder + '/' + pkl_folder, true_group_struct)
    ## all best group structures
    ResAll.bestGroupStructures()
    ## For a given model, compile the rank of amiable group structures
    ## for each simulation. The table looks like:
    ##
    ##              sim1    sim2    sim3 ...
    ##  gstruct 1
    ##  gstruct 2
    ##  gstruct 4
    ##  ...
    ##
    ## Also, print the mean and standard deviation for the ranks of each
    ## amiable group structure.
    amiable_ranks_all = ResAll.rankOfAmiableGroupStructuresInTrainingWithoutPenalty(printing=True)
    ## Percentage of times for each (mu, alpha) that the true group structure
    ## is selected during training phase
    selection_freq_true = ResAll.selectionFreqOfTrueGroupStructureForEachParameter(printing=True)
    ## List for all models the Percentage of times for each (mu, alpha)
    ## that the true group structure is selected
    true_gstructs_dict = {1: Group([1, 2], [3, 4], [5, 6]),
                          2: Group([1], [2, 3], [4, 5, 6]),
                          3: Group([1], [2], [3], [4], [5], [6]),
                          4: Group([1, 2, 3, 4, 5, 6]),
                          5: Group([1, 3], [2], [4, 5, 6]),
                          6: Group([1, 2, 3, 4], [5, 6])}

    selection_freq_dict = defaultdict(list)
    for model_id in [1, 2, 3, 4, 5, 6]:
        sim_folder = "/home/panc/research/OKGT/software/okgtreg/okgtreg/simulation/sim_02052016"
        pkl_folder = "validate_exhaustive_model" + str(model_id)
        ResAll = ValidateExhaustiveProcessorAll(sim_folder + '/' + pkl_folder,
                                                true_gstructs_dict[model_id])
        for k, v in ResAll.selectionFreqOfTrueGroupStructureForEachParameter().iteritems():
            selection_freq_dict[k].append(v)

    selection_freq_dict_orderByKey = OrderedDict(sorted(selection_freq_dict.items()))
    ## Print latex table
    for k, v in selection_freq_dict_orderByKey.iteritems():
        print "{0:10} & {1:5} & {2:5} & {3:5} & {4:5} & {5:5} & {6:5} & {7:5} \\\\".format(
            "%.4e" % (k[0]), "%.02f" % (k[1]), v[0], v[1], v[2], v[3], v[4], v[5])


        # Extract a list of the test R2 for all (mu, alpha)
        # then make a plot
