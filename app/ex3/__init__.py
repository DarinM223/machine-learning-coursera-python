import scipy.io
import os

DIGIT_PATH = os.path.join(os.path.dirname(__file__), 'data1.mat')

digit_data = scipy.io.loadmat(DIGIT_PATH)

