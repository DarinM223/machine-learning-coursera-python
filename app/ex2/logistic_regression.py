import scipy as sp
from scipy.optimize import fmin

def sigmoid(z):
    """
    Calculates the sigmoid of z (1/(1 + e^-z))
    """
    return 1/(1.0 + sp.exp(-z))

def costFunction(theta, X, y):
    """
    Calculate the cost and gradient for logistic regression
    """
    m = sp.shape(y)[0]
    J = (1.0/m) * ((-y).T.dot(sp.log(sigmoid(X.dot(theta)))) - \
                   (-y + 1).T.dot(sp.log(-sigmoid(X.dot(theta)) + 1)))
    grad = (1.0/m) * (sigmoid(X.dot(theta)) - y).T.dot(X)
    return (J, m)

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using the learned 
    logistic regression parameters theta
    """
    return sigmoid(X.dot(theta)) >= 0.5

def find_minimum_theta(theta, X, y):
    """
    Finds the minimum theta using the bfgs algorithm
    Retuns the minimum theta and the value of the cost function
    for that theta
    """
    result = fmin(lambda theta, X, y: costFunction(theta, X, y)[0], \
                  x0=theta, args=(X, y), maxiter=400, full_output=True)
    return result[0], result[1]

