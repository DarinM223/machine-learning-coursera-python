import scipy as sp
import os

PATH = os.path.join(os.path.dirname(__file__), 'data1.txt') 

admission_data = sp.genfromtxt(PATH, delimiter=',')

