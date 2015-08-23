import scipy as sp

from app.ex2 import admission_data
from logistic_regression import costFunction, costFunctionReg, predict, find_minimum_theta

def run():
    theta = sp.zeros((3, 1))
    data = sp.copy(admission_data)
    X = data[:, [0, 1]]
    y = data[:, [2]]
    m = sp.shape(y)[0]

    # Add intercept term to x
    X = sp.concatenate((sp.ones((m, 1)), X), axis=1)

    (theta, _) =  find_minimum_theta(theta, X, y)

    (J, grad) = costFunction(theta, X, y)
    print 'Cost: ',J[0, 0]
    print 'Gradient: ',grad
    (J, grad) = costFunctionReg(theta, X, y, 0.1)
    print 'Cost: ',J[0, 0]
    print 'Gradient: ',grad

