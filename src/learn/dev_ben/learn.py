import glob
import sys
import os
import numpy as np
from os.path import dirname, abspath

from src import Utils
Utils.set_keras_backend("tensorflow")

from keras.models import Sequential
from keras.layers import Dense

project_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
training_data_dir = os.path.join(project_dir, 'data/training_data')
project_root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
training_data_dir = os.path.join(project_root_dir, 'data/training_data')
csv_files = glob.glob(os.path.join(training_data_dir, '*'))
if len(csv_files) is 0:
    print('no files for training found')
    sys.exit(1)

X = []
Y = []

for i, path in enumerate(csv_files):
    filename = os.path.basename(path)
    data = np.genfromtxt(path, dtype=float, delimiter=';')
    print('importing ' + str(len(data)) + ' lines from ' + filename + ' (' + str(i+1) + '/' + str(len(csv_files)) + ')')

    for line in data:
        x = line[:-2]
        y = [0 for _i in range(82)]  # pos zero is PASS, so we need 1+81
        move_idx = int(line[81])
        if move_idx is -1:  # = PASS
            y[0] = 1
        else:
            y[move_idx + 1] = 1
        X.append(x)
        Y.append(y)

# it might be better to append to a numpy array right away?
# np.concatenate would do the job, but seems very memory costly to reassign it in a loop?
# a = np.array([[1,2,3]]) b = np.array([[4,5,6]]) c = np.concatenate((a,b))
X = np.array(X)
Y = np.array(Y)


# SET UP NETWORK TOPOLOGY
model = Sequential()
model.add(Dense(162, input_dim=81, activation='relu'))  # first parameter of Dense is number of neurons
model.add(Dense(82, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAIN
model.fit(X, Y, epochs=20)

# EVALUATE
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# STORE TRAINED NN
# model.save('model.h5')
# via keras.io/getting-started/faq/#how-can-i-save-a-keras-model
json_file = open('model_architecture.json', 'w')
json_file.write(model.to_json())
json_file.close()
model.save_weights('model_weights.h5')

# from keras.utils import plot_model
# requires pydot and graphviz to be installed
# plot_model(model, to_file='model.png', show_shapes=True)
