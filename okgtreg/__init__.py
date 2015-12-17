"""
1. Post discussing how to organize a python package with multiple files (modules):
    http://stackoverflow.com/questions/12540290/how-to-organize-multiple-python-files-into-a-single-module-without-it-behaving-l

"""

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Data import Data, ParameterizedData
from okgtreg.Group import Group, RandomGroup
from okgtreg.Kernel import Kernel
from okgtreg.OKGTReg import OKGTReg, OKGTRegForDetermineGroupStructure
from Parameters import Parameters
from DataUtils import readSkillCraft1
