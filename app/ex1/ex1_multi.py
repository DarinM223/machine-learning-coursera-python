import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from app.ex1 import housing_data, gradient_descent as graddesc

def run():
    data = sp.copy(housing_data)
    x = data[:, [0, 1]]
    y = data[:, [2]]
    m = sp.shape(y)[0]
    
    # Normalize the x values
    (x, mu, sigma) = graddesc.featureNormalize(x)
    
    # Add intercept term to x
    x = sp.concatenate((sp.ones((m, 1)), x), axis=1)
    
    # Init Theta and run Gradient Descent
    num_iters = 400
    
    # Choose some alpha value
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
    
    for alpha in alphas:
        theta = sp.zeros((3, 1))
        (theta, J_history) = graddesc.gradientDescent(x, y, theta, alpha, num_iters)
        # Plot the value of J by number of iterations
        plt.plot(range(1, J_history.size+1), J_history, '-b')
        plt.title('Alpha = %f' % (alpha))
        plt.xlabel('Number of iterations')
        plt.ylabel('J')
        plt.xlim([0, 50])
        plt.show(block=True)
    
        # Estimate the price of a 1650 sq-ft, 3 br house
        price = 0
        house = sp.array([[1.0, 1650.0, 3.0]])
        # Normalize the features
        house[0, 1:] = (house[0, 1:] - mu) / sigma
        price = house.dot(theta)
        print('The estimated price with alpha', alpha, 'is', price[0, 0])
    
    # Reload the data
    data = sp.copy(housing_data)
    
    x = data[:, [0, 1]]
    y = data[:, [2]]
    
    # Add intercept term to x
    x = sp.concatenate((sp.ones((m, 1)), x), axis=1)
    
    # Calculate the normal equation
    theta = graddesc.normalEqn(x, y)
    print('Theta computed from the normal equations:')
    print(theta)
