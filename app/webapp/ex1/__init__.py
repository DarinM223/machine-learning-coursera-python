import scipy as sp
from app.ex1 import housing_data
from app.ex1.gradient_descent import computeCost, featureNormalize, gradientDescent, normalEqn

data = sp.copy(housing_data)
x = data[:, [0, 1]]
y = data[:, [2]]
m = sp.shape(y)[0]

# Normalize the x values
(x, mu, sigma) = featureNormalize(x)

# Add intercept term to x
x = sp.concatenate((sp.ones((m, 1)), x), axis=1)

# Init Theta and run Gradient Descent
num_iters = 400

# Choose some alpha value
alpha = 0.3

# Calculate theta
theta = sp.zeros((3, 1))
(theta, J_history) = gradientDescent(x, y, theta, alpha, num_iters)

normal_theta = sp.zeros((3, 1))
normal_theta = normalEqn(x, y)
