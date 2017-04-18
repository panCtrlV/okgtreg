__author__ = 'panc'

'''
Prepare data for the response Murders:

 1. Extract the most related covariates (23) from screening
 2. Remove missing values
 3. Convert data types to float
'''

import numpy as np
import pickle

from okgtreg.DataUtils import readCrimeData_unnormalize
from okgtreg.Data import Data

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

########################################
# Read the screening result (pkl file) #
# for murders                          #
########################################
pkl_folder = "okgtreg/application/crime_unnormalize_02262016/screening"
pkl_file = "crime_unnormalize_screening-murders-20160226-233849-262260.pkl"
file_path = pkl_folder + '/' + pkl_file
with open(file_path, 'rb') as f:
    screen_res = pickle.load(f)

# Kept attributes after screening (OKGT R2 > 0.99)
attr_names = [k for k, v in screen_res['screen_keep'].items() if v > 0.99]
counter = 0
for name in attr_names:
    counter += 1
    print '[%d]' % counter, name

# Prepare data with reduced attributes
## Extract related variables
crimeAfterScreenDF = crimeDF[attr_names + ['murders']]
## Remove rows with missing values
valueIdx_setList = []
for c in crimeAfterScreenDF.columns:
    print c
    if crimeAfterScreenDF[c].dtype == object:
        valueIdx = crimeAfterScreenDF[c][crimeAfterScreenDF[c] != '?'].index
        valueIdx_setList.append(set(valueIdx))
    valueIdx = list(reduce(lambda x, y: x & y, valueIdx_setList))

crimeAfterScreenCleanDF = crimeAfterScreenDF.ix[valueIdx, :].astype(float)
# for c in crimeAfterScreenCleanDF:
#     print crimeAfterScreenCleanDF[c].dtype
y = np.array(crimeAfterScreenCleanDF['murders'])
x = np.array(crimeAfterScreenCleanDF[attr_names])

## Create Data object
data = Data(y, x)
data.setYName('murders')
data.setXNames(attr_names)

###############
# Pickle data #
###############
filename = "okgtreg/application/crime_unnormalize_02262016/cleanDataForMurders.pkl"
with open(filename, 'wb') as f:
    pickle.dump(data, f)
