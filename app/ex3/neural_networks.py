import scipy as sp
import matplotlib as plt
from app.ex2 import logistic_regression as logres

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    a1 = sp.c_[sp.ones((m, 1)), X]
    z2 = Theta1.dot(a1.T)
    a2 = logres.sigmoid(z2)
    a2 = sp.r_[sp.ones((1, a2.shape[1])), a2]

    # Output layer
    z3 = Theta2.dot(a2)
    a3 = logres.sigmoid(z3)

    result = []
    for i in range(0, m):
        prediction = sp.argmax(a3[:, i]) + 1
        result.append(prediction)
    return result

