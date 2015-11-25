from okgtreg.DataSimulator import DataSimulator



"""
We start with a random partition of the predictor variables. The corresponding
OKGT is fitted with R2 being recorded. Then perform the following split and join
operators:

1. For each group of size > 1, split it into individual covariates and fit
   the corresponding OKGT and record its R2.

   Compare R2 under each scenario in (2) with the R2 we started with. Pick the
   the scenario with gives the largest improvement in R2 and randomly pick a
   covariate from the group to form a uni-variate group.

2. For each univariate group in the current structure, try to merge its covariate
   with one of the other groups, regardless of the other group being univariate or
   multivariate.

Then, step 1 and 2 are repeated iteratively until no further improvement in R2.

Can be extended to include removing steps for variable selection.

Question:

1. If different random partitions, do they result in the same optimal group structure?
"""

# Simulate data
data = DataSimulator.SimData_Wang04(100)

# Random partition to start with
group0 = Rando
