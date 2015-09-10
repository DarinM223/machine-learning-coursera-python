import scipy as sp
import matplotlib as plt

from app.ex3 import digit_data, weights_data, multi_classification as multiclass, neural_networks as neurnet

def run():
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    print('Loading and Visualizing Data...')

    X, y = digit_data['X'], digit_data['y']
    m = X.shape[0]

    # Randomly select 100 data points to display
    sel = sp.random.permutation(m)

    multiclass.displayData(X[sel, :])

    print('Program paused. Press enter to continue.')
    raw_input()

    Theta1, Theta2 = weights_data['Theta1'], weights_data['Theta2']
    neurnet.predict(Theta1, Theta2, X)

    print('Program paused. Press enter to continue.')
    raw_input()

    rp = sp.random.permutation(m)
    for i in range(0, m):
        pred = neurnet.predict(Theta1, Theta2, sp.asmatrix(X[rp[i], :]))
        print("Neural Network Prediction: %d (digit %d)\n" % (pred[0], sp.mod(pred[0], 10)))

