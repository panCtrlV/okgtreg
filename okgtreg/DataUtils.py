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
