import scipy as sp
from scipy.optimize import fmin

def sigmoid(z):
    """
    Calculates the sigmoid of z (1/(1 + e^-z))
    """
    return 1/(1.0 + sp.exp(-z))

def costFunction(flattendTheta, X, y):
    """
    Calculate the cost and gradient for logistic regression
    """
    # numpy fmin function only allows flattened arrays instead of
    # matrixes which is stupid so it has to be converted every time
    theta = sp.asmatrix(flattendTheta).T
    m = sp.shape(y)[0]
    J = (1.0/m) * ((-y).T.dot(sp.log(sigmoid(X.dot(theta)))) - \
                   (-y + 1).T.dot(sp.log(-sigmoid(X.dot(theta)) + 1)))
    grad = (1.0/m) * (sigmoid(X.dot(theta)) - y).T.dot(X)
    return (J, grad)

def costFunctionReg(flattendTheta, X, y, lmbda):
    """
    Calculate the cost and gradient for logistic regression
    using regularization (helps with preventing overfitting
    with many features)
    """
    # numpy fmin function only allows flattened arrays instead of
    # matrixes which is stupid so it has to be converted every time
    theta = sp.asmatrix(flattendTheta).T
    m = sp.shape(y)[0]
    (J, grad) = costFunction(flattendTheta, X, y)

    # f is a filter vector that will disregard regularization for theta0
    f = sp.ones((3, 1))
    f[0, 0] = 0
    thetaFiltered = sp.multiply(theta, f)

    J = J + (lmbda/(2.0 * m)) * (thetaFiltered.T.dot(thetaFiltered))
    grad = grad + ((lmbda/m) * thetaFiltered).T

    return (J, grad)

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using the learned 
    logistic regression parameters theta
    """
    return sigmoid(X.dot(theta)) >= 0.5

def _costFn(theta, X, y):
    result = costFunction(theta, X, y)
    return result[0][0]

def find_minimum_theta(theta, X, y):
    """
    Finds the minimum theta 
    Returns the minimum theta and the value of the cost function
    for that theta
    """
    result = fmin(_costFn, x0=theta, args=(X, y), maxiter=400, full_output=True)
    return result[0], result[1]

