This folder contains more simulation study on the effect of group structures on OKGT fitting. This is a continuation
study of "sim_01172016" and "sim_01192016". 

In this study, we still keep the number of variables small in each model ($<=6$). Various group structures and transformations are tested. The following models are used in the study;

1. Linear model:

    $$ln(Y) = \beta_0 + \beta_1*X_1 + \beta_2*X_2 + \ldots + \beta_6*X_6$$

2. Single index model:

    $$\exp(Y) = (\beta_0 + \beta_1*X_1 + \beta_2*X_2 + \ldots + \beta_6*X_6)**2$$

3. Partial linear single index model:

    $$Y^{1/2} = 1 + \alpha_1*X_1 + \alpha_2*X_2 + \alpha_3*X_3 + \sin(\beta_1*X_4 + \beta_2*X_5 + \beta_3*X_6)$$
    $$Y^{1/2} = 1 + \alpha_1*X_1 + \alpha_2*X_2 + \alpha_3*X_3 + sigmoid(\beta_1*X_4 + \beta_2*X_5 + \beta_3*X_6)$$
    $$Y^3 = 1 + \alpha_1*X_1 + \alpha_2*X_2 + \alpha_3*X_3 + \exp(\beta_1*X_4 + \beta_2*X_5 + \beta_3*X_6)$$
    $$Y^{1/3} = \alpha_0 + \alpha_1*X_1 + \alpha_2*X_2 + \alpha_3*X_3 + \ln(|\beta_1*X_4 + \beta_2*X_5 + \beta_3*X_6|)$$
    $$logit(Y) = \alpha_0 + \alpha_1*X_1 + \alpha_2*X_2 + \alpha_3*X_3 + |\beta_1*X_4 + \beta_2*X_5 + \beta_3*X_6|$$
    
4. Additive model:

    $$ln(Y) = \beta_0 + \beta_1*X_1 + \beta_2*X_2^2 + \beta_3*X_3^3 + \beta_4*|X_4| + \beta_5*\sin(X_5) + \beta_6*\exp{X_6}$$

5. Optimal group transformation:

    $$\exp(Y) = 1 + np.abs(\alpha + sin(X_1) + X_2 * X_3 + sigmoid(X_4 * X_5 * X_6))$$
    
    $$ln(Y) = \alpha + X_1^2 + |X_2 * X_3| + 2 * sigmoid(X_4 * X_5 * X_6)$$
    
    $$ln(Y) = \alpha + \exp(X_1) + (X_2 * X_3)_+ + 2 * sin(X_4 * X_5 * X_6)$$
    
    $$\exp(Y) = 1 + ln(X_1) + X_2 / \exp(X_3) + (X_4 + X_5)^{X_6}$$
    
    
    