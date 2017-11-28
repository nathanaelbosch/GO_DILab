"""Learn from the training data"""
import random as rn
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout

DIR = 'data/training_data/simplest_move_prediction'
DEV_PATH = 'src/learn/dev_nath_win_prediction'
np.random.seed(123)  # for reproducibility


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# 1. Get data
def list_all_csv_files(dir):
    """List all sgf-files in a dir

    Recursively explores a path and returns the filepaths
    for all files ending in .sgf
    """
    root_dir = os.path.abspath(dir)
    sgf_files = []
    for root, sub_dirs, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)
            name, extension = os.path.splitext(path)
            if extension == '.csv':
                sgf_files.append(path)
    return sgf_files


def main():
    # all_files = list_all_csv_files(DIR)
    # all_files = rn.sample(all_files, 1000)
    file = os.path.join(DEV_PATH, '5000_games.csv')

    in_size = 2*9*9
    # data = pd.read_csv(file, sep=',', header=None)
    data = np.genfromtxt(file, delimiter=',')
    print(data.shape)
    X = data[:, :(in_size)]
    y = data[:, in_size:]

    with open(os.path.join(DEV_PATH, 'mean_var.txt'), 'w') as f:
        f.write(str(np.mean(X)))
        f.write('\n')
        X -= np.mean(X)
        # print(np.std(X))
        f.write(str(np.std(X)))
        f.write('\n')
        X /= np.std(X)

    # split into 67% for train and 33% for test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    print(X_train[0].shape)
    print(y_train[0].shape)

    # create model
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    model = Sequential()
    model.add(Dense(100, input_dim=in_dim, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(100, input_dim=in_dim, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(100, input_dim=in_dim, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(100, input_dim=in_dim, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(out_dim, activation='sigmoid'))

    # compile model
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, epochs=10, batch_size=1000)

    # Evaluate model
    scores = model.evaluate(X_test, y_test)
    print('\n{:s}: {:.2f}'.format(model.metrics_names[1], scores[1]*100))

    model.save(os.path.join(DEV_PATH, 'model.h5'))


if __name__ == '__main__':
    main()
