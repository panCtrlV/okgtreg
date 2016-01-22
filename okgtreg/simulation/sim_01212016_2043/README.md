Re-run simulation for those "hard" models in "sim_01202016_1636". This time, laplace kernel is used.

The followings are the hard models in "sim_01202016_1636" that are used in the current study:


Model 2
=======

$$y = \ln((1 + x'b + e)^2)$$


Model 5
=======

$$y^3 = 1 + b_1 * x_1 + b_2 * x_2 + b_3 * x_3 + \exp(b_4 * x_4 + b_5 * x_5 + b_6 * x_6) + e$$

Model 6
=======

$$y = \ln(|1 + b_1 * x_1 + b_2 * x_2 + b_3 * x_3 + \ln(|b_4 * x_4 + b_5 * x_5 + b_6 * x_6|) + e|)$$


Model 7
=======

$$y = sigmoid(1 + b_1 * x_1 + b_2 * x_2 + b_3 * x_3 + |b_4 * x_4 + b_5 * x_5 + b_6 * x_6| + e)$$