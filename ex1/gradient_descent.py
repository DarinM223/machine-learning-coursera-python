import scipy as sp
import scipy.linalg as linalg

def featureNormalize(x):
    """
    Returns the normalized feature by subtracting the mean 
    and dividing by the standard deviation
    """
    mu = sp.zeros((1, sp.shape(x)[1]))
    sigma = sp.zeros((1, sp.shape(x)[1]))

    mu = sp.mean(x, axis=0)
    sigma = sp.std(x, axis=0)
    t = sp.ones((len(x), 1))

    x_norm = (x - mu) / sigma
    return (x_norm, mu, sigma)

def computeCost(x, y, theta):
    """
    Computes the cost function for multiple parameters
    """
    m = sp.shape(y)[0]
    J = 1/(2.0 * m) * (x.dot(theta) - y).T.dot(x.dot(theta) - y)
    return J

def gradientDescent(x, y, theta, alpha, num_iters):
    """
    Computes the gradient descent for multiple parameters
    """
    m = sp.shape(y)[0]
    J_history = sp.zeros((num_iters, 1))

    # Copy vector so that you don't change existing one
    grad = sp.copy(theta)
    alpha_div_m = alpha / m

    for i in range(0, num_iters):
        inner_sum = (((x.dot(grad) - y).T).dot(x)).T
        grad = grad - alpha_div_m * inner_sum
        J_history[i] = computeCost(x, y, grad)

    return (grad, J_history)

def normalEqn(x, y):
    theta = linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return theta
