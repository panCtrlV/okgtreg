__author__ = 'panc'

from okgtreg.Parameters import Parameters
from okgtreg.Data import ParameterizedData


def predict(test_data, kernel, gstruct, fns_est):
    parameters = Parameters(gstruct, kernel, [kernel] * gstruct.size)
    parameterizedTestData = ParameterizedData(test_data, parameters)
    test_ghat_list = [fns_est[i](parameterizedTestData.getXFromGroup(i))
                      for i in range(1, gstruct.size + 1)]
    test_ghat = sum(test_ghat_list) + fns_est[0]  # with intercept
    return test_ghat


def predictionError(g, ghat):
    return sum((g - ghat) ** 2) / len(g)
