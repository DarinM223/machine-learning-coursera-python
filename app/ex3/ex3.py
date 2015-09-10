import scipy as sp
import matplotlib as plt

from app.ex3 import digit_data, multi_classification as multiclass

def run():
    X, y = digit_data['X'], digit_data['y']
    m = X.shape[0]

    rand_indices = sp.random.permutation(m)

    multiclass.displayData(X)

    print('Program paused. Press enter to continue.')
    raw_input()

    print('Training One-vs-All Logistic Regression...')
    lamda = 0.1
    num_labels = 10
    all_theta = multiclass.oneVsAll(X, y, num_labels, lamda)

    print('Program paused. Press enter to continue.')
    raw_input()

    print all_theta.shape

    multiclass.predict(all_theta, X, y)

