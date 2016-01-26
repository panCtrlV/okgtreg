# Simulation study

Including penalized $R^2$ in forward inclusion/selection method with penalty to determine the best group structure for a data set. The penalty is in the form of $\sum_{\ell=1}^d \lambda p_{\ell}^{p_{\ell}}$, where $d$ is the number of groups in the group structure, $\ell \in \{1, \ldots, d\}$ is the group index, $p_\ell$ is the number of variables in $\ell$-th group, and $\lambda$ is a tuning parameter. The selection criteria for the optimal group structure is to minimize the following penalized mean squared error of OKGT with respect to the group structure:

$$ 
\frac{1}{n} \| \left( g(\boldsymbol{y}) - \sum_{\ell=1}^d f_\ell(\boldsymbol{x}_\ell)  \|^2 - \lambda p_{\ell}^{p_{\ell}}.
$$
 
We used the same 10 models used in "sim_01202016_1636" with the only exception that the standard t distribution in Model 6 is replaced by the standard Normal distribution. 

For each model, 100 simulations are run, with 500 sample size for each time. 

The results for each model are saved as a dictionary where the Group object is the key, and the value of the penalized R^2 is the value.

