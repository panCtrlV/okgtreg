__author__ = 'panc'

'''
Screen predictor variables for each response.

There are 18 response variables in the data set and
125 predictor variables.
'''
import sys
import numpy as np
import operator

from okgtreg.DataUtils import readCrimeData_unnormalize
from okgtreg.Data import Data
from okgtreg.Kernel import Kernel
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.Group import Group

args = sys.argv
respNameId = int(args[1])  # 1 ~ 18
# respNameId = 3

###########################################################
# Read data as a Pandas DataFrame.                        #
# Since there are missing values at different positions   #
# in each column, the data needs to be further processed. #
###########################################################
crimeDF = readCrimeData_unnormalize()

# For each response variable, construct a Data object
#   including the predictor variables (attributes with
#   predictive power noted on UCI webpage)
predictorNames = ['population', 'householdSize',
                  'racePctBlack', 'racePctWhite', 'racePctAsian', 'racePctHisp',
                  'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
                  'numbUrban', 'pctUrban',
                  'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec',
                  'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap',
                  'blackPerCap', 'indianPerCap', 'asianPerCap', 'otherPerCap', 'hispPerCap',
                  'numUnderPov', 'pctPopUnderPov',
                  'pctLess9thGrade', 'pctNotHSGrad', 'pctBSorMore',
                  'pctUnemployed', 'pctEmploy',
                  'pctEmplManu', 'pctEmplProfServ', 'pctOccupManu', 'pctOccupMgmtProf',
                  'malePctDivorce', 'malePctNevMarr', 'femalePctDiv', 'totalPctDiv',
                  'persPerFam', 'pctFam2Par', 'pctKids2Par', 'pctYoungKids2Par', 'pctTeen2Par',
                  'pctWorkMomYoungKids', 'pctWorkMom', 'numKidsBornNeverMar', 'pctKidsBornNeverMar',
                  'numImmig', 'pctImmigRecent', 'pctImmigRec5', 'pctImmigRec8', 'pctImmigRec10',
                  'pctRecentImmig', 'pctRecImmig5', 'pctRecImmig8', 'pctRecImmig10',
                  'pctSpeakEnglOnly', 'pctNotSpeakEnglWell',
                  'pctLargHouseFam', 'pctLargHouseOccup', 'persPerOccupHous', 'ersPerOwnOccHous',
                  'persPerRentOccHous', 'pctPersOwnOccup', 'pctPersDenseHous',
                  'pctHousLess3BR', 'medNumBR', 'housVacant', 'pctHousOccup', 'pctHousOwnOcc',
                  'pctVacantBoarded', 'pctVacMore6Mos', 'medYrHousBuilt', 'pctHousNoPhone',
                  'pctWOFullPlumb',
                  'ownOccLowQuart', 'ownOccMedVal', 'ownOccHiQuart', 'ownOccQrange',
                  'rentLowQ', 'rentMedian', 'rentHighQ', 'rentQrange', 'medRent', 'medRentPctHousInc',
                  'medOwnCostPctInc', 'medOwnCostPctIncNoMtg',
                  'numInShelters', 'numStreet',
                  'pctForeignBorn', 'pctBornSameState',
                  'pctSameHouse85', 'pctSameCity85', 'pctSameState85',
                  'lemasSwornFT', 'lemasSwFTPerPop', 'lemasSwFTFieldOps', 'lemasSwFTFieldPerPop',
                  'lemasTotalReq', 'lemasTotReqPerPop', 'policReqPerOffic', 'policPerPop',
                  'racialMatchCommPol', 'pctPolicWhite', 'pctPolicBlack', 'pctPolicHisp',
                  'pctPolicAsian', 'pctPolicMinor',
                  'officAssgnDrugUnits', 'numKindsDrugsSeiz', 'policAveOTWorked',
                  'landArea',
                  'popDens',
                  'pctUsePubTrans',
                  'policCars', 'policOperBudg', 'lemasPctPolicOnPatr', 'lemasGangUnitDeploy',
                  'lemasPctOfficDrugUn', 'policBudgPerPop']

responseNames = ['murders', 'murdPerPop',
                 'rapes', 'rapesPerPop',
                 'robberie', 'robbbPerPop',
                 'assaults', 'assaultPerPop',
                 'burglaries', 'burglPerPop',
                 'larcenies', 'larcPerPop',
                 'autoTheft', 'autoTheftPerPop',
                 'arsons', 'arsonsPerPop',
                 'violentCrimesPerPop',
                 'nonViolPerPop']

##############################################################
# For a given response variable,                             #
#   fitting a marginal OKGT for each predictor variables,    #
#   where a linear kernel is equipped to the response and    #
#   Gaussian (sigma=0.5) is used for the predictor variable. #
#   Then, screening is preformed by selecting predictors     #
#   with high values of R2.                                  #
##############################################################
screen_r2_dict = {}  # estimated R2
screen_n_dict = {}  # sample size
screen_nComp_dict = {}  # number of Nystroem components

# respName = responseNames[0]
respName = responseNames[respNameId - 1]
ySeries = crimeDF[respName]
# In case, y has missing values, we remove the
#   corresponding indices.
if ySeries.dtype == object:
    hasMissingInResponse = True
    yValueIdx = ySeries[ySeries != '?'].index  # index of non-missing response values
    ySeries = ySeries[yValueIdx].astype(float)
else:
    hasMissingInResponse = False
print "Response variable: ", respName, '\n'
print "{0:5} {1:25} {2:5} {3:5} {4:15}".format('id', 'predictor', 'n', 'nComp', 'R2')
counter = 0
for predName in predictorNames:
    counter += 1
    # print predName, ':', crimeDF[predName].dtype
    print "{0:5} {1:25}".format("[%d]" % counter, predName),
    # if it is an 'object' type, then there are missing values
    xSeries = crimeDF[predName]
    # remove the samples corresponding to the missing y, if there is any
    if hasMissingInResponse:
        xSeries = xSeries[yValueIdx]
    # remove the samples corresponding to the missing x, if there is any
    if xSeries.dtype == object:
        # Reference: find element's index in pandas Series
        #   http://stackoverflow.com/questions/18327624/find-elements-index-in-pandas-series
        valueIdx = xSeries[xSeries != '?'].index  # index of non-missing
        xCleanSeries = xSeries[valueIdx].astype(float)
        yCleanSeries = ySeries[valueIdx]
        x = np.array(xCleanSeries)
        y = np.array(yCleanSeries)
    elif np.issubdtype(xSeries.dtype, int):
        # if integer, covert to float
        xSeries = xSeries.astype(float)
        x = np.array(xSeries)
        y = np.array(ySeries)
    else:
        x = np.array(xSeries)
        y = np.array(ySeries)

    # A data set with one predictor
    data = Data(y, x[:, np.newaxis])
    data.setXNames([predName])
    # TODO: allow a string as a name if there is only
    # todo: predictor variable
    data.setYName(respName)
    ## record the sample size used for this predictor variable
    screen_n_dict[predName] = data.n
    print "{0:5}".format(data.n),

    # Fit OKGT
    kernel = Kernel('gaussian', sigma=0.5)
    okgt = OKGTReg2(data, kernel=kernel, group=Group([1]))
    nComp = int(np.ceil(0.1 * data.n))  # for Nystroem nComponent, depend on sample size
    ## record the number of components used for Nystroem method
    screen_nComp_dict[predName] = nComp
    print "{0:5}".format(nComp),
    margFit = okgt.train('nystroem', nComp, 25)  # use low-rank approximation
    ## record the estimated R2
    screen_r2_dict[predName] = margFit['r2']
    print "{0:15}".format("%.10f" % margFit['r2'])

# //////////////////////////////////////////////////////////////
# It is suspected that a smaller sample size
#   is associated with a higher R2. That is
#   why during the screening, we set nComp
#   proportional to the number of non-missing values.
#
# import matplotlib.pyplot as plt
# plt.scatter(screen_n_dict.values(), screen_r2_dict.values())
#
## From the plot, the estimated R2 seems less
##  affected by the sample size.
# ////////////////////////////////////////////////////////////////


####################################################
# Thresholding R2s to keep the predictor variables #
#   with high R2 values.                           #
####################################################
screen_keepAttr_dict = {}  # retained attributes
threshold = 0.80
print "=== Attributes with R2 >", threshold, 'for predicting', respName, 'after screening ==='
counter = 0
for k, v in screen_r2_dict.iteritems():
    if v > threshold:
        counter += 1
        screen_keepAttr_dict[k] = v

# print in decreasing order
sorted_keepAttr = sorted(screen_keepAttr_dict.iteritems(),
                         key=operator.itemgetter(1), reverse=True)

print "{0:4} {1:25} {2:15}".format('rank', 'predictor', 'R2')
counter = 0
for k, v in sorted_keepAttr:
    counter += 1
    print "{0:4} {1:25} {2:15}".format(counter, k, "%.10f" % v)

####################################
# Pickle / Save the screen results #
####################################
dump_dict = dict(screen_n=screen_n_dict,
                 screen_nComp=screen_nComp_dict,
                 screen_r2=screen_r2_dict,
                 screen_keep=screen_keepAttr_dict)

import os
import pickle
from okgtreg.utility import currentTimestamp

timestamp = currentTimestamp()

filename, file_extension = os.path.splitext(__file__)
filename = filename + "-" + \
           respName + "-" + \
           timestamp + ".pkl"
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dump_dict, f)
