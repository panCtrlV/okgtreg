__author__ = 'panc'

import platform
import pickle

from okgtreg.Parameters import Parameters
from okgtreg.Data import ParameterizedData


def readCleanDataForMurders():
    my_platform = platform.system()
    if my_platform == 'Linux':
        okgtreg_folder = "/home/panc/research/OKGT/software/okgtreg"
    elif my_platform == 'Darwin':
        okgtreg_folder = "/Users/panc25/Dropbox/Research/Zhu_Michael/my_paper/paper_OKGT/software/okgtreg"
    else:
        raise NotImplementedError("** Platform System Cannot be Recognized! **")

    pkl_file_path = okgtreg_folder + '/' + "okgtreg/application/crime_unnormalize_02262016/cleanDataForMurders.pkl"
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def predict(test_data, kernel, gstruct, fns_est):
    parameters = Parameters(gstruct, kernel, [kernel] * gstruct.size)
    parameterizedTestData = ParameterizedData(test_data, parameters)
    test_ghat_list = [fns_est[i](parameterizedTestData.getXFromGroup(i))
                      for i in range(1, gstruct.size + 1)]
    test_ghat = sum(test_ghat_list) + fns_est[0]  # with intercept
    return test_ghat


def predictionError(g, ghat):
    return sum((g - ghat) ** 2) / len(g)
