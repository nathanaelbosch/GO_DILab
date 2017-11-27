import os
import numpy as np
from os.path import dirname, abspath

from src import Utils
Utils.set_keras_backend("tensorflow")

from keras.models import Sequential
from keras.layers import Dense


data_dir = os.path.join(dirname(dirname(dirname(dirname(abspath(__file__))))), 'data')
training_data_dir = os.path.join(data_dir, 'training_data')

X = np.array([])
Y = np.array([])


for csv_file in os.listdir(training_data_dir):
    if csv_file != 'game_100672.sgf.csv': continue  # dev restriction

    data = np.genfromtxt(os.path.join(training_data_dir, csv_file), dtype=float, delimiter=';')

    # TODO


# X = np.array([
#     inp,
# ])
# Y = np.array([
#     outp,
# ])
#
#
# # set up network topology
# model = Sequential()
# dim = rows * cols
# # first arg of Dense is # of neurons
# model.add(Dense(162, input_dim=dim, activation='relu'))
# # last layer = output layer, must have 81 again
# model.add(Dense(dim, activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, Y, epochs=1)
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#
# print(model.predict(X))
