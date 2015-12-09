"""
Test SkillCraft1 data set reading method
"""
from okgtreg.DataUtils import readSkillCraft1

data = readSkillCraft1()
data
print data

"""
Test Data.__repr__
"""
from okgtreg.DataUtils import readSkillCraft1

import pandas as pd

scDF = pd.read_csv('okgtreg/data/SkillCraft1.csv')
names = scDF.columns.values
names

data = readSkillCraft1()
# Set name for the response
data.setYName(names[1])
# Set names for the covariates
data.setXNames(names[2:])

# Fail to set names for the covariates
data.setXNames(names)
data.setXNames(names[5:])

# Access name attributes
data.xnames
data.yname

# Pretty Data object
print data
data

# Access data by name
data['Age']
data['LeagueIndex']
data['ComplexUnitsMade']

# No variable names
from okgtreg.DataSimulator import DataSimulator

data = DataSimulator.SimData_Wang04WithInteraction2(50)
print data