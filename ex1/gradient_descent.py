import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt

sp.seterr(invalid='warn')

def drawPlot(data):
    """
    Draws a scatter plot of the data plot
    """
    x = data[:, 0]
    y = data[:, 1]
    m = len(y)
    
    plt.scatter(x, y, marker='.')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.show()

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

PATH = './data.txt'

data = sp.genfromtxt(PATH, delimiter=',')

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
alphas = [0.01, 0.03, 0.1, 0.3, 1.0]

for alpha in alphas:
    theta = sp.zeros((3, 1))
    (theta, J_history) = gradientDescent(x, y, theta, alpha, num_iters)
    # Plot the value of
    numel = J_history.size
    plt.plot(range(1, J_history.size+1), J_history, '-b')
    plt.title("Alpha = %f" % (alpha))
    plt.xlabel('Number of iterations')
    plt.ylabel('J')
    plt.xlim([0, 50])
    plt.show(block=True)

# Reload the data
data = sp.genfromtxt(PATH, delimiter=',')

x = data[:, [0, 1]]
y = data[:, [2]]

# Add intercept term to x
x = sp.concatenate((sp.ones((m, 1)), x), axis=1)

# Calculate the normal equation
theta = normalEqn(x, y)
print("Theta computed from the normal equations:")
print(theta)
