import scipy as sp
from scipy.optimize import fmin
import matplotlib.pyplot as plt

def sigmoid(z):
    """Calculates the sigmoid of z (1/(1 + e^-z))"""
    # return 1/(1.0 + sp.exp(-z))
    return sp.special.expit(z)

def costFunction(flattendTheta, X, y):
    """Calculate the cost and gradient for logistic regression"""
    # numpy fmin function only allows flattened arrays instead of
    # matrixes which is stupid so it has to be converted every time
    flattendTheta = sp.asmatrix(flattendTheta)
    (a, b) = flattendTheta.shape
    if a < b:
        theta = flattendTheta.T
    else:
        theta = flattendTheta
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
    flattendTheta = sp.asmatrix(flattendTheta)
    (a, b) = flattendTheta.shape
    if a < b:
        theta = flattendTheta.T
    else:
        theta = flattendTheta
    m = sp.shape(y)[0]
    (J, grad) = costFunction(theta, X, y)

    # f is a filter vector that will disregard regularization for theta0
    f = sp.ones((theta.shape[0], 1))
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

def _costFnReg(theta, X, y, lmbda):
    result = costFunctionReg(theta, X, y, lmbda)
    return result[0][0]

def find_minimum_theta(theta, X, y):
    """
    Finds the minimum theta 
    Returns the minimum theta and the value of the cost function
    for that theta
    """
    result = fmin(_costFn, x0=theta, args=(X, y), maxiter=400, full_output=True)
    return result[0], result[1]

def find_minimum_theta_reg(theta, X, y, lmbda):
    """
    Finds the minimum theta using the BFGS algorithm
    Returns the minimum theta, and the value of the cost function for that theta
    """
    result = sp.optimize.minimize(_costFnReg, theta, args=(X, y, lmbda), method='BFGS', options={
        'maxiter': 500,
        'disp': True
    })
    return result.x, result.fun

def mapFeature(X1, X2):
    """Feature mapping function to polynomial features"""
    degree = 6
    out = sp.ones((sp.shape(X1)[0], 1))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            term1 = X1 ** (i - j)
            term2 = X2 ** (j)
            term = (term1 * term2).reshape(sp.shape(term1)[0], 1)
            out = sp.hstack((out, term))
    return out

def plotData(data):
    """Plots students that are both admitted and not admitted into a university"""
    pos = data[data[:, 2] == 1]
    neg = data[data[:, 2] == 0]

    plt.scatter(pos[:, 0], pos[:, 1], c='b', marker='+', s=40, linewidths=2)
    plt.scatter(neg[:, 0], neg[:, 1], c='y', marker='o', s=40, linewidths=1)

def plotDecisionBoundary(data, X, theta):
    """
    Plots the data points X and y into a new figure with the
    decision boundary defined by theta
    """
    plotData(data)
    if X.shape[1] <= 3:
        plot_x = sp.array([min(X[:, 1]), max(X[:, 1])])
        plot_y = (-1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y)
        plt.axis([30, 100, 30, 100])
    else:
        u = sp.linspace(-1, 1.5, 50)
        v = sp.linspace(-1, 1.5, 50)
        z = sp.zeros((len(u), len(v)))

        for i in range(0, len(u)):
            for j in range(0, len(v)):
                z[i, j] = (mapFeature(sp.array([u[i]]), sp.array([v[j]]))).dot(theta)

        z = z.T # important to transpose z before calling contour

        u, v = sp.meshgrid(u, v)

        plt.contour(u, v, z, [0.0, 0.0])

