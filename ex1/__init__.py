import gradient_descent
import scipy as sp
import os

PATH = os.path.join(os.path.dirname(__file__), 'data.txt')

housing_data = sp.genfromtxt(PATH, delimiter=',')
