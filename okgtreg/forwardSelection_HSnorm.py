"""
Instead of using R2 as the selection criterion, this implementation
uses HS norm of the cross-covariance operator and joint cross-covariance
operator.

Motivated by R2 formula:

    R2 = Cov(Y, \hat{Y}) / Var(Y),

we can define the non-parametric version of R2 for OKGT as

    R2 = Cov( g(Y), \sum_j f_j(X_j) ) / Var(g(Y)).  --- (1)

The calculation of the covariance and variance can be done by using
cross-covariance operators as;

    Cov(g(Y), \sum_j {f_j(X_j)})
        = \sum_j Cov( g(Y), f_j(X_j) )
        = \sum_j < g, R_{YX_j}f_j >_{H_Y},      --- (2)

    Var(g(Y)) = < g, R_{YY}g >_{H_Y}.       --- (3)

If we use block operator notation, (2) could be simplified to:

    Cov(g(Y), \sum_j {f_j(X_j)}) = < g(Y), R^+_{YX} (f_1, f_2, ..., f_p) >_{H_Y}        --- (4)

where R^+_{YX} is the column stack of R_{YX_j}'s.

...
"""

# Calculate covariance operator R_{YY}

# Calculate cross-covariance operator R^+_{YX}



