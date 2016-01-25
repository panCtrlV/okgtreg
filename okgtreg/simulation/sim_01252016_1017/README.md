# Simulation study

Including penalized $R^2$ in forward inclusion/selection method to determine the best group structure for a data set.
 
We used the same 10 models used in "sim_01202016_1636" with the only exception that the standard t distribution in Model 6 is replaced by the standard Normal distribution. 

For each model, 100 simulations are run, with 500 sample size for each time. 

The results for each model are saved as a dictionary where the Group object is the key, and the value of the penalized R^2 is the value.

