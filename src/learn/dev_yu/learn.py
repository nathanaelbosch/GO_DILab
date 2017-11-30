import os
from os.path import dirname, abspath
import numpy as np
from numpy import genfromtxt

from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
np.random.seed(100)

project_root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
data_dir = os.path.join(project_root_dir, 'data')
training_set_dir = os.path.join(data_dir, 'training_set')
csv_files = [os.path.join(training_set_dir, 'some_game.sgf.csv')]


# 1. Load data
for path in csv_files:
    dataset = genfromtxt(path, delimiter=';')

#input X, output Y
X = np.column_stack((dataset[:,0:81],dataset[:,82]))
