import scipy as sp
import matplotlib.pyplot as plt
import app.ex2.logistic_regression as logres

def computeCost( theta, X, y, lamda ):
    """Copied cost function because fmin_cg messes up the dimensions"""
    m = sp.shape( X )[0]
    hypo = logres.sigmoid( X.dot( theta ) )
    term1 = sp.log( hypo ).dot( -y )
    term2 = sp.log( 1.0 - hypo ).dot( 1 - y )
    left_hand = (term1 - term2) / m
    right_hand = theta.T.dot( theta ) * lamda / (2*m)
    return left_hand + right_hand

def gradientCost( theta, X, y, lamda ):
    """Copied gradient function because fmin_cg messes up the dimensions"""
    m = sp.shape( X )[0]
    grad = X.T.dot( logres.sigmoid( X.dot( theta ) ) - y ) / m
    grad[1:] = grad[1:] + ( (theta[1:] * lamda ) / m )
    return grad

def oneVsAll( X, y, num_classes, lamda ):
    """Copied multi classification because numpy sucks"""
    m,n = sp.shape( X )
    X = sp.c_[sp.ones((m, 1)), X]
    all_theta = sp.zeros((n+1, num_classes))
    
    for k in range(0, num_classes):
        theta = sp.zeros(( n+1, 1 )).reshape(-1)
        temp_y = ((y == (k+1)) + 0).reshape(-1)
        result = sp.optimize.fmin_cg( computeCost, fprime=gradientCost, x0=theta, \
                args=(X, temp_y, lamda), maxiter=50, disp=False, full_output=True )
        all_theta[:, k] = result[0]
        print "%d Cost: %.5f" % (k+1, result[1])
    return all_theta

def predict( theta, X, y ):
    """
    Predict the label for a trained one-vs-all classifier
    Copied because it is impossible to port properly from matlab to numpy
    """
    m,n = sp.shape( X )
    X = sp.c_[sp.ones((m, 1)), X]

    correct = 0
    for i in range(0, m ):
        prediction = sp.argmax(theta.T.dot( X[i] )) + 1
        actual = y[i]
        if actual == prediction:
            correct += 1
    print "Accuracy: %.2f%%" % (correct * 100.0 / m )

def displayData(X, theta = None):
    """Display 2D data in a nice grid"""
    width = 20
    rows, cols = 10, 10
    out = sp.zeros((width * rows, width * cols))

    rand_indices = sp.random.permutation(5000)[0:rows * cols]

    counter = 0
    for y in range(0, rows):
        for x in range(0, cols):
            start_x = x * width
            start_y = y * width
            out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
            counter += 1

    img = sp.misc.toimage(out)
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.imshow(img)

    if theta is not None:
        result_matrix = []
        X_biased = sp.c_[sp.ones(X.shape[0]), X]

        for idx in rand_indices:
            result = (sp.argmax(theta.T.dot(X_biased[idx])) + 1) % 10
            result_matrix.append(result)
        result_matrix = sp.array(result_matrix).reshape(rows, cols).transpose()
        print(result_matrix)

    plt.show()

