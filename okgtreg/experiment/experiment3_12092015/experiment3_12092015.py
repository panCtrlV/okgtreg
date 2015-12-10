# ---
# The following two lines added the root of the package into the system path
# so that the module okgtreg can be imported properly when you run the script
# from the project root directory, i.e. top level okgtreg.

# import sys
# sys.path.append('../okgtreg')

# Alternatively, you can add the package directory to PYTHONPATH. That is, in bash
#
#    export PYTHONPATH=/your/path/to/okgtreg/package:$PYTHONPATH
#
# Then, you can run the script from anywhere.
# ---

"""
Test group structure detection on SkillCraft1 data set.
"""
import time

from okgtreg.DataUtils import readSkillCraft1
from okgtreg.Kernel import Kernel


data = readSkillCraft1()
kernel = Kernel('gaussian', sigma=0.5)

# # Forward
# from okgtreg.groupStructureDetection.forwardSelection import forwardSelection
#
# start_time = time.time()
# res_forward = forwardSelection(data, kernel, True, 10, seed=25)
# elapsed_time = time.time() - start_time
# print "Elapsed time: ", elapsed_time

# # Backward
# from okgtreg.groupStructureDetection.backwardStructureDetermination import backwardSelection
#
# start_time = time.time()
# res_backward = backwardSelection(data, kernel, True, 10, seed=25)
# elapsed_time = time.time() - start_time
# print "Elapsed time: ", elapsed_time

# Split and merge
from okgtreg.groupStructureDetection.splitAndMergeWithRandomInitial import splitAndMergeWithRandomInitial2

start_time = time.time()
res_splitAndMerge = splitAndMergeWithRandomInitial2(data, kernel, True, 10, seed=25)
elapsed_time = time.time() - start_time
print "Elapsed time: ", elapsed_time
