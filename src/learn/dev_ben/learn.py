import os
import numpy as np
from os.path import dirname, abspath

from src import Utils
Utils.set_keras_backend("tensorflow")

from keras.models import Sequential
from keras.layers import Dense


data_dir = os.path.join(dirname(dirname(dirname(dirname(abspath(__file__))))), 'data')
training_data_dir = os.path.join(data_dir, 'training_data')

X = []
Y = []

for csv_file in os.listdir(training_data_dir):
    if csv_file != 'game_100672.sgf.csv': continue  # dev restriction

    data = np.genfromtxt(os.path.join(training_data_dir, csv_file), dtype=float, delimiter=';')

    for line in data:
        x = line[:-2]
        y = [0 for _i in range(82)]  # pos 0 is PASS
        if line[81] is -1:  # = PASS
            y[0] = 1
        else:
            y[int(line[81]) + 1] = 1
        X.append(x)
        Y.append(y)

# it might be (much) better to append to a numpy array right away?
# np.concatenate would do the job, but seems very memory costly to reassign it in a loop?
# a = np.array([[1,2,3]]) b = np.array([[4,5,6]]) c = np.concatenate((a,b))
X = np.array(X)
Y = np.array(Y)

# set up network topology
model = Sequential()
# first arg of Dense is # of neurons
model.add(Dense(162, input_dim=81, activation='relu'))
model.add(Dense(82, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=1)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print(model.predict(X))
