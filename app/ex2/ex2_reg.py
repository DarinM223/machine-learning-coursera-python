import scipy as sp
import matplotlib.pyplot as plt

from app.ex2 import microchip_data, logistic_regression as logres

def run():
    theta = sp.zeros((3, 1))
    data = sp.copy(microchip_data)
    X = data[:, [0, 1]]
    y = data[:, [2]]
    m = sp.shape(y)[0]

    logres.plotData(data)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.show()

    """
    Regularized Logistic Regression
    """

    X = logres.mapFeature(data[:, 0], data[:, 1])

    initial_theta = sp.zeros((X.shape[1], 1))

    lmbda = 1

    (J, grad) = logres.costFunctionReg(initial_theta, X, y, lmbda)

    print('Cost at initial theta (zeros): ', J[0,0])
    print('Program paused. Press enter to continue.')
    raw_input()

    """
    Regularization and Accuracies
    """

    initial_theta = sp.zeros((X.shape[1], 1))
    lmbda = 1

    (theta, J) = logres.find_minimum_theta_reg(initial_theta, X, y, lmbda)

    logres.plotDecisionBoundary(data, X, theta)
    plt.legend(['y = 1', 'y = 0', 'Decision Boundary'])
    plt.show()

