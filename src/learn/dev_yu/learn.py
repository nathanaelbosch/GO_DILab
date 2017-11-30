import os
from os.path import dirname, abspath
import numpy as np
from numpy import genfromtxt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

# fix random seed for reproducibility
np.random.seed(100)

project_root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
data_dir = os.path.join(project_root_dir, 'data')
training_set_dir = os.path.join(data_dir, 'training_set')
csv_file = os.path.join(training_set_dir, 'some_game.sgf.csv')


# 1. Load data
dataset = genfromtxt(csv_file, delimiter=';')

# input X (board position and current player), output Y (next move)
X = np.column_stack((dataset[:,0:81],dataset[:,82]))
Y = to_categorical(dataset[:,81], num_classes=82)  # one-hot encoding, class 82 stands for pass


# 2. Define model
model = Sequential()
model.add(Dense(200, kernel_initializer='normal', bias_initializer='normal', input_dim=82, activation='relu'))
model.add(Dense(200, kernel_initializer='normal', bias_initializer='normal', activation='relu'))
model.add(Dense(82, activation='softmax'))  # softmax guarantees a probability distribution over the 81 locations and pass


# 3. Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 4. Fit model
model.fit(X, Y, epochs=50, batch_size=50)


# 5. Evaluate model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#6. Make predictions
# pred = model.predict(X)