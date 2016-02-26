"""
The utility functions for reading and processing data sets
"""

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
    scDF = pd.read_csv('okgtreg/data/SkillCraft1.csv')
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


def readHousingData():
    # Reference: delimiter
    # http://stackoverflow.com/questions/15026698/how-to-make-separator-in-read-csv-more-flexible-wrt-whitespace
    # No missing value in this data set
    houseDF = pd.read_csv('okgtreg/data/housing/housing.data', header=None, delimiter='\s+')
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAZ',
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
