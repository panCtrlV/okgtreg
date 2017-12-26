__author__ = 'panc'

'''
Since we already have the results from 1-fold validation
simulation where the test R2 is not penalized. We can calculate
the penalized test R2's by using these results. So we don't
need simulation.

In particular, we can use the following two object:

    selectedGroupStructuresForEachParameter_dic
    testedGroupStructures_dict

For each (mu, alpha), it is associated with group structure selected
from the training phase. Then, the group structure can be mapped to
the test R2 without penalty. Thus the penalized test R2 can be calucated
for each (mu, alpha).
'''
import glob
from collections import defaultdict, OrderedDict

from okgtreg.Group import Group

from okgtreg.groupStructureDetection.backwardPartition import rkhsCapacity
from okgtreg.simulation.sim_02052016.validate_exhaustive_analyze import ValidateExhaustiveProcessor, \
    ValidateExhaustiveProcessorAll


class ValidateExhaustiveProcessor_PenalizeTestR2(ValidateExhaustiveProcessor):
    def bestParametersByPenalizingTestR2(self, ordered=True):
        select_dict = self.res['select']
        test_dict = self.res['test']
        # Calculate the penalized test R2 for each (mu, alpha)
        penalizedTestR2ForEachParameter_dict = {}
        for k, v in select_dict.iteritems():
            penalty = k[0] * rkhsCapacity(Group(group_struct_string=v), k[1])
            penalizedR2 = test_dict[v] - penalty
            penalizedTestR2ForEachParameter_dict[k] = penalizedR2
        if ordered:
            return OrderedDict(sorted(penalizedTestR2ForEachParameter_dict.items()))
        else:
            return penalizedTestR2ForEachParameter_dict


if __name__ == '__main__':
    # Unpickle a .pkl file
    model_id = 1
    sim_folder = "/home/panc/research/OKGT/software/okgtreg/okgtreg/simulation/sim_02052016"
    pkl_folder = "validate_exhaustive_model" + str(model_id)
    file_name = "validate_exhaustive-model1-seed100-201602190127.pkl"
    true_group_struct = Group([1, 2], [3, 4], [5, 6])  # model 1

    Res = ValidateExhaustiveProcessor(pkl_file_path=sim_folder + '/' + pkl_folder + '/' + file_name,
                                      true_group_struct=true_group_struct)

    # Using one .pkl file
    # Calculate the penalized test R2 for each (mu, alpha)
    Res2 = ValidateExhaustiveProcessor_PenalizeTestR2(
        pkl_file_path=sim_folder + '/' + pkl_folder + '/' + file_name,
        true_group_struct=true_group_struct)

    Res2.bestParametersByPenalizingTestR2()

    # Using all .pkl files
    # Get the best parameters from each .pkl file
    Res2_list = []
    for fpath in glob.glob(pkl_folder + '/' + "*.pkl"):
        Res2_list.append(ValidateExhaustiveProcessor_PenalizeTestR2(pkl_file_path=fpath,
                                                                    true_group_struct=true_group_struct))

    bestParameters_all_list = []
    for Res in Res2_list:
        bestParameters_all_list.append(Res.bestParametersByPenalizingTestR2().items()[0])

    for param, r2 in bestParameters_all_list:
        print param, ":", r2
