import scipy as sp
import os

ADMISSION_PATH = os.path.join(os.path.dirname(__file__), 'data1.txt') 
MICROCHIP_PATH = os.path.join(os.path.dirname(__file__), 'data2.txt')

admission_data = sp.genfromtxt(ADMISSION_PATH, delimiter=',')
microchip_data = sp.genfromtxt(MICROCHIP_PATH, delimiter=',')

