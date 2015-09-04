import scipy as sp
import matplotlib.pyplot as plt

from app.ex2 import admission_data, logistic_regression as logres

def run():
    theta = sp.zeros((3, 1))
    data = sp.copy(admission_data)
    X = data[:, [0, 1]]
    y = data[:, [2]]
    m = sp.shape(y)[0]

    # Add intercept term to x
    X = sp.concatenate((sp.ones((m, 1)), X), axis=1)

    """
    Part 1: Plotting
    """

    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
    logres.plotData(data)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend('Admitted', 'Not admitted')
    plt.show()

    print('Program paused. Press enter to continue.')
    raw_input()

    """
    Part 2: Compute Cost and Gradient
    """

    (m, n) = X.shape

    initial_theta = sp.zeros((n, 1))

    (cost, grad) = logres.costFunction(initial_theta, X, y)

    print('Cost at initial theta (zeros): ', cost)
    print('Gradient at initial theta (zeros): ', grad)

    print('Program paused. Press enter to continue.')
    raw_input()

    """
    Part 3: Optimizing using fminunc
    """

    (theta, cost) = logres.find_minimum_theta(theta, X, y)

    print('Cost at theta found by fmin: ', cost)
    print('Theta: ', theta)

    logres.plotDecisionBoundary(data, X, theta)

    plt.show()

    """
    Part 4: Predict and Accuracies
    """

    prob = logres.sigmoid(sp.asmatrix([1, 45, 85]).dot(theta))
    print('For a student with scores 45 and 85, we predict an admission probability of ', prob[0, 0])
    print('Program paused. Press enter to continue.')

