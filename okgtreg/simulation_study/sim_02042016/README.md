In this simulation study, we run exhaustive algorithm for group structure identification. 

Since the exhaustive algorithm is suitable for OKGT with small number of covariates, we use models with six covariates. In order to evaluate the performance of the penalized OKGT method on different models, we will use different group structure specifications. In particular, the following group structures will be used for the simulation study:

1. $\{ [1], [2], [3], [4], [5], [6] \}$
2. $\{ [1,2], [3,4], [5,6] \}$
3. $\{ [1,2,3], [4,5,6]\}$
4. $\{ [1,2,3,4,5,6] \}

The models are:

1. $y^{1/2} = x_1 + x_2^2 + x_3^3 + \sin(x_4) + \ln(x_5) + |x_6| + \epsilon$
2. $y^{1/2} = (x_1 + x_2)^2 + \ln(x_3^2 + x_4^2) + x_5^x_6 + \epsilon$
3. $y^{1/2} = (x_1 + x_2 + x_3)^2 + \ln(x_4^2 + x_5^2 + x_6^2) + \epsilon$
4. $y^{1/2} = np.ln(\| x \|^2)$

Since the penalized OKGT is used for group structure determination, we need to select the values for the tuning parameters $\mu$ and $a$. We fix $\mu=1e-4$ which gives a relatively stable results in the previous simulation studies. Then we choose $a$ in the range of [1, 10]. 

The simulated data set is split into 500 for training and 200 for testing. For each model, the value of $a$ is chosen such that the selected model from training phase performed best on the testing set. The model the trained by using penalized OKGT, while the testing step uses regular OKGT to obtain the goodness of fit. 

