import sqlite3
import random

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.utils.data as Data

import src.learn.bots.utils as utils
from .model import ConvNet


DB_PATH = 'data/db.sqlite'


class Learn():
    def __init__(self, **kwargs):
        np.random.seed(1234)
        self.training_size = kwargs.get('training_size', 1000)
        self.data_retrieval_command = '''
            SELECT *
            FROM elo_ordered_games
            ORDER BY RANDOM()
            LIMIT ?'''
        self.db = sqlite3.connect(DB_PATH)

        self.epochs = kwargs.get('epochs', 5)
        self.batch_size = kwargs.get('batch_size', 100)
        self.print_every = kwargs.get('print_every',
            int(self.training_size*0.8/self.batch_size))
        self.conv_depth = kwargs.get('conv_depth', 5)
        self.no_cuda = kwargs.get('no_cuda', False)
        self.symmetries = kwargs.get('symmetries', False)
        self.test = kwargs.get('test', True)

    def get_data(self):
        data = pd.read_sql_query(
            self.data_retrieval_command,
            self.db,
            params=[self.training_size])
        return data

    def format_data(self, data):
        boards = data[data.columns[3:-2]].as_matrix()

        passes = data['move'] == -1
        y = data['move']
        y[passes] = 81
        y = y.values
        X = utils.encode_board(boards, data['color'])
        X = X.reshape(-1, 3, 9, 9).astype(float)

        X = X[~passes]
        y = y[~passes]
        assert X.shape[0] == y.shape[0], 'Something went wrong with the shapes'
        assert (X.sum(axis=0).sum(axis=0) == X.shape[0]).all()
        print('X.shape:', X.shape, 'X.dtype:', X.dtype)
        print('y.shape:', y.shape, 'y.dtype:', y.dtype)

        example = random.choice(range(X.shape[0]))
        print('Example input:', X[example])
        print('Example output:', y[example])

        return X, y

    @staticmethod
    def transform(board, move, rot, flip):
        rot = rot % 360
        if flip:
            board = np.fliplr(board)
            move = np.fliplr(move)
        for i in range(0, rot, 90):
            board = np.rot90(board, axes=(2, 3))
            move = np.rot90(move, axes=(1, 2))
        return board, move

    @staticmethod
    def label_to_board(labels):
        n_entries = labels.shape[0]
        boards = np.zeros((n_entries, 81))
        boards[np.arange(n_entries), labels] = 1
        boards = boards.reshape(-1, 9, 9)
        return boards

    @staticmethod
    def board_to_label(boards):
        boards = boards.reshape(-1, 81)
        labels = np.argmax(boards, axis=1)
        return labels

    def train_model(self):
        model = self.model
        X, y = self.X_train, self.y_train
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        if not self.no_cuda and torch.cuda.is_available():
            print('Using cuda')
            model.cuda()
            # criterion.cuda()

        dataset = Data.TensorDataset(data_tensor=X, target_tensor=y)
        data_loader = Data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        print('Start training')
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(data_loader, 0):

                # Handle symmetries
                _inputs, _labels = data
                _inp, _lab = _inputs.numpy(), _labels.numpy()
                _syms = 8 if self.symmetries else 1
                for j in range(_syms):

                    optimizer.zero_grad()

                    moves = self.label_to_board(_lab)
                    inputs, moves = self.transform(
                        _inp, moves,
                        rot=j*90, flip=j>=360)
                    labels = self.board_to_label(moves)
                    inputs, labels = inputs.copy(), labels.copy()
                    inputs, labels, _, _ = self.to_pytorch(
                        inputs, labels, inputs, labels)

                    if not self.no_cuda and torch.cuda.is_available():
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # print('Inputs in cuda:', inputs.is_cuda)
                    # print('Model in cutorch.from_numpy(boards.copy()))da:', next(model.parameters()).is_cuda)
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()

                    optimizer.step()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.data).sum()
                    accuracy = (100 * correct / total)

                    running_loss += loss.data[0]

                # print statistics
                if i % self.print_every == self.print_every - 1:
                    print('[{:d}, {:5d}]  loss: {:.3f}  accuracy: {:.3f}%'.
                          format(epoch + 1, i + 1,
                                 running_loss / self.print_every,
                                 accuracy))
                    running_loss = 0.0
                    correct = 0
                    total = 0
            if self.test:
                self.test_model()
        print('Finished Training')

    def to_pytorch(self, X_train, y_train, X_test=None, y_test=None):
        if X_test is not None and y_test is not None:
            return (
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train),
                torch.from_numpy(X_test).float(),
                torch.from_numpy(y_test))
        else:
            return (
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train))

    def train_test_split(self, X, y, p=0.9):
        msk = np.random.rand(X.shape[0]) < 0.9
        return X[msk], y[msk], X[~msk], y[~msk]

    def test_model(self):
        dataset = Data.TensorDataset(
            data_tensor=self.X_test, target_tensor=self.y_test)
        test_loader = Data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data
            if not self.no_cuda and torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('Test Accuracy: {:.3f}%'.format(
            100 * correct / total))

    def run(self):
        data = self.get_data()
        X, y = self.format_data(data)

        if self.test:
            X_train, y_train, X_test, y_test = self.train_test_split(X, y)
            X_train, y_train, X_test, y_test = self.to_pytorch(
                X_train, y_train, X_test, y_test)
            self.X_train, self.y_train, self.X_test, self.y_test = (
                X_train, y_train, X_test, y_test)
        else:
            self.X_train, self.y_train = self.to_pytorch(X, y)

        self.model = ConvNet(self.X_train.size(), self.y_train.size())
        self.train_model()
        if self.test:
            self.test_model()
        print('Save model')
        torch.save(self.model, 'src/learn/conv/convnet.pt')


def test():
    _l = Learn(
        training_size=10,
        batch_size=1,
        # print_every=1,
        epochs=3,
        conv_depth=2,
        no_cuda=True,
    )
    _l.data_retrieval_command = '''
        SELECT *
        FROM elo_ordered_games
        LIMIT ?'''
    _l.run()


def overfit():
    _l = Learn(
        training_size=10,
        batch_size=1,
        epochs=1000,
        conv_depth=2,
        no_cuda=True,
        test=False,
        symmetries=True,
    )
    _l.data_retrieval_command = '''
        SELECT *
        FROM elo_ordered_games
        LIMIT ?'''
    _l.run()


def main():
    Learn(
        training_size=1000000,
        batch_size=3000,
        print_every=50,
        epochs=5000,
        conv_depth=2,
        # symmetries=True,
    ).run()


if __name__ == '__main__':
    main()
    # test()
    # overfit()
