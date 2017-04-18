"""
The utility functions for reading and processing data sets
"""
import platform
import pandas as pd
import numpy as np

from okgtreg.Data import Data


def readSkillCraft1():
    """
    Pre-process and read SkillCraft1 data set into memory by:
        1) removing rows with missing values;
        2) converting data type to float

    The first column of "GameID" is removed.

    :rtype: Data
    :return:
    """
    scDF = pd.read_csv('okgtreg/data/skillCraft1/SkillCraft1.csv')
    n1 = len(scDF.index)
    names = scDF.columns.values
    # Remove rows with missing values
    # and covert data type to float
    for c in scDF.columns:
        if scDF[c].dtype == object:
            scDF = scDF[scDF[c] != '?']
            scDF[c] = scDF[c].astype(float)
    n2 = len(scDF.index)
    print("** Rows with missing values are removed. %d => %d. **" % (n1, n2))

    # Separate response and predictors
    y = scDF['LeagueIndex']
    x = scDF.ix[:, 2:]

    scData = Data(np.array(y), np.array(x))
    scData.setYName(names[1])
    scData.setXNames(names[2:])

    return scData


def readCrimeData():
    crimeDF = pd.read_csv('okgtreg/data/crime/communities.data', header=None)
    n1 = len(crimeDF.index)
    names = crimeDF.columns.values  # integers as header names
    # Remove rows with missing values
    for c in crimeDF.columns:
        if crimeDF[c].dtype == object:
            crimeDF = crimeDF[crimeDF[c] != '?']
    n2 = len(crimeDF.index)
    pass


def readCrimeData_unnormalize():
    my_platform = platform.system()
    if my_platform == 'Linux':
        okgtreg_folder = "/home/panc/research/OKGT/software/okgtreg"
    elif my_platform == 'Darwin':
        okgtreg_folder = "/Users/panc25/Dropbox/Research/Zhu_Michael/my_paper/paper_OKGT/software/okgtreg"
    else:
        raise NotImplementedError("** Platform System Cannot be Recognized! **")

    file_path = okgtreg_folder + '/' + 'okgtreg/data/crime_unnormalize/CommViolPredUnnormalizedData.txt'
    crimeDF = pd.read_csv(file_path, header=None)
    names = ['communityName', 'state', 'countyCode', 'communityCode', 'fold',
             'population', 'householdSize',
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
             'lemasPctOfficDrugUn', 'policBudgPerPop',
             'murders', 'murdPerPop',
             'rapes', 'rapesPerPop',
             'robberie', 'robbbPerPop',
             'assaults', 'assaultPerPop',
             'burglaries', 'burglPerPop',
             'larcenies', 'larcPerPop',
             'autoTheft', 'autoTheftPerPop',
             'arsons', 'arsonsPerPop',
             'violentCrimesPerPop',
             'nonViolPerPop']
    crimeDF.columns = names
    print("**[Warning] there are missing values, marked by \'?\' **")
    # TODO: enable multivariate response in Data class
    return crimeDF


def readHousingData():
    # Reference: delimiter
    # http://stackoverflow.com/questions/15026698/how-to-make-separator-in-read-csv-more-flexible-wrt-whitespace
    # No missing value in this data set
    my_platform = platform.system()
    if my_platform == 'Linux':
        filepath = "/home/panc/research/OKGT/software/okgtreg/okgtreg/data/housing/housing.data"
    elif my_platform == 'Darwin':
        filepath = "okgtreg/data/housing/housing.data"
    else:
        raise NotImplementedError("** Platform System Cannot be Recognized! **")

    houseDF = pd.read_csv(filepath, header=None, delimiter='\s+')
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
             'PTRATIO', 'B', 'LSTAT', 'MEDV']
    houseDF.columns = names
    # Separate response and predictors
    y = houseDF['MEDV']
    x = houseDF.ix[:, 0:13]
    houseData = Data(np.array(y), np.array(x))
    houseData.setYName(names[-1])
    houseData.setXNames(names[:-1])
    return houseData


if __name__ == '__main__':
    houseData = readHousingData()
