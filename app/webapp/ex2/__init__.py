import scipy as sp
from app.ex2 import admission_data
from app.ex2.logistic_regression import find_minimum_theta

data = sp.copy(admission_data)
X = data[:, [0, 1]]
y = data[:, [2]]
m = sp.shape(y)[0]

# Add intercept term to x
X = sp.concatenate((sp.ones((m, 1)), X), axis=1)

theta = sp.zeros((3, 1))
(theta, _) = find_minimum_theta(theta, X, y)

