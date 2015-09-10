import scipy.io
import os

DIGIT_PATH = os.path.join(os.path.dirname(__file__), 'data1.mat')
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights.mat')

digit_data = scipy.io.loadmat(DIGIT_PATH)
weights_data = scipy.io.loadmat(WEIGHTS_PATH)

