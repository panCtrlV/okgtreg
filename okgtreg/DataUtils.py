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
    sc_df = pd.read_csv('okgtreg/data/SkillCraft1.csv')
    # Remove rows with missing values
    # and covert data type to float
    for c in sc_df.columns:
        if sc_df[c].dtype == object:
            sc_df = sc_df[sc_df[c] != '?']
            sc_df[c] = sc_df[c].astype(float)

    # Separate response and predictors
    y = sc_df['LeagueIndex']
    x = sc_df.ix[:, 2:]

    return Data(np.array(y), np.array(x))
