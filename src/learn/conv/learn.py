import os
import sqlite3

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.utils.data as Data


import src.learn.bots.utils as utils


DB_PATH = 'data/db.sqlite'


class ConvNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, conv_depth=5):
        super(ConvNet, self).__init__()

        _, in_channels, in1, in2 = input_dim
        n_filters = 128

        self.start_conv = torch.nn.Conv2d(
            in_channels,
            out_channels=n_filters,
            kernel_size=5,
            padding=2)
        self.start_relu = torch.nn.ReLU()

        self.mid_convs = []
        self.relus = []
        for i in range(conv_depth):
            self.mid_convs.append(torch.nn.Conv2d(
                n_filters,
                out_channels=n_filters,
                kernel_size=3,
                padding=1))
            self.relus.append(torch.nn.ReLU())

        self.last_conv = torch.nn.Conv2d(
            n_filters,
            out_channels=1,
            kernel_size=1)

        self.end_relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(128*9*9, 9*9+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self._forward(x)

    def _forward_with_symmetries(self, x):
        """DISREGARD FOR NOW

        Compute symmetries at runtime, evaluate the model for each, then
        average over the predictions and transform symmetries back
        """
        _, c, w, h = x.size()
        x_np = x.data.numpy()
        boards = x_np

        # Create symmetries
        boards_90 = np.rot90(boards, axes=(2, 3))
        boards_180 = np.rot90(boards, k=2, axes=(2, 3))
        boards_270 = np.rot90(boards, k=3, axes=(2, 3))
        boards_flipped = np.fliplr(boards)
        boards_flipped_90 = np.rot90(np.fliplr(boards), axes=(2, 3))
        boards_flipped_180 = np.rot90(np.fliplr(boards), k=2, axes=(2, 3))
        boards_flipped_270 = np.rot90(np.fliplr(boards), k=3, axes=(2, 3))

        # Symmetries to pytorch
        boards = Variable(torch.from_numpy(boards.copy()))
        boards_90 = Variable(torch.from_numpy(boards_90.copy()))
        boards_180 = Variable(torch.from_numpy(boards_180.copy()))
        boards_270 = Variable(torch.from_numpy(boards_270.copy()))
        boards_flipped = Variable(torch.from_numpy(boards_flipped.copy()))
        boards_flipped_90 = Variable(torch.from_numpy(boards_flipped_90.copy()))
        boards_flipped_180 = Variable(torch.from_numpy(boards_flipped_180.copy()))
        boards_flipped_270 = Variable(torch.from_numpy(boards_flipped_270.copy()))

        if x.is_cuda:
            boards.cuda()
            boards_90.cuda()
            boards_180.cuda()
            boards_270.cuda()
            boards_flipped.cuda()
            boards_flipped_90.cuda()
            boards_flipped_180.cuda()
            boards_flipped_270.cuda()

        out = self._forward(boards)
        out_90 = self._forward(boards_90)
        out_180 = self._forward(boards_180)
        out_270 = self._forward(boards_270)
        out_flipped = self._forward(boards_flipped)
        out_flipped_90 = self._forward(boards_flipped_90)
        out_flipped_180 = self._forward(boards_flipped_180)
        out_flipped_270 = self._forward(boards_flipped_270)

        return None

    def _forward(self, x):
        x = self.start_conv(x)
        x = self.start_relu(x)
        for conv, relu in zip(self.mid_convs, self.relus):
            x = conv(x)
            x = relu(x)
        # x = self.last_conv(x)
        # x = self.softmax(x)
        x = x.view(-1, 128*9*9)

        x = self.fc(x)
        x = self.softmax(x)
        return x

    def cuda(self):
        super(ConvNet, self).cuda()
        for conv in self.mid_convs:
            conv.cuda()


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

        self.print_every = kwargs.get('print_every', 1000)
        self.epochs = kwargs.get('epochs', 5)
        self.batch_size = kwargs.get('batch_size', 100)
        self.conv_depth = kwargs.get('conv_depth', 5)

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
        print('X.shape:', X.shape, 'X.dtype:', X.dtype)
        print('y.shape:', y.shape, 'y.dtype:', y.dtype)

        return X, y

    def train_model(self):
        model = self.model
        X, y = self.X_train, self.y_train
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        if torch.cuda.is_available():
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
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # print('Inputs in cuda:', inputs.is_cuda)
                # print('Model in cuda:', next(model.parameters()).is_cuda)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum()
                accuracy = (100 * correct / total)

                # print statistics
                running_loss += loss.data[0]
                if i % self.print_every == self.print_every-1:
                    print('[{:d}, {:5d}]  loss: {:.3f}  accuracy: {:.3f}%'.format(epoch + 1, i + 1, running_loss / 2000, accuracy))
                    running_loss = 0.0
                    correct = 0
                    total = 0
            self.test_model()
        print('Finished Training')

    def to_pytorch(self, X_train, y_train, X_test, y_test):
        return (
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train),
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test))

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
            if torch.cuda.is_available():
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

        X_train, y_train, X_test, y_test = self.train_test_split(X, y)
        X_train, y_train, X_test, y_test = self.to_pytorch(
            X_train, y_train, X_test, y_test)

        self.X_train, self.y_train, self.X_test, self.y_test = (
            X_train, y_train, X_test, y_test)
        self.model = ConvNet(X_train.size(), y_test.size())
        self.train_model()
        self.test_model()
        print('Save model')
        torch.save(self.model, 'src/learn/conv/convnet.pt')


def test():
    _l = Learn(
        training_size=100,
        batch_size=10,
        print_every=1,
        epochs=3,
        conv_depth=1,
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
        print_every=1,
        epochs=100,
        conv_depth=1,
    )
    _l.data_retrieval_command = '''
        SELECT *
        FROM elo_ordered_games
        LIMIT ?'''
    _l.run()


def main():
    Learn(
        training_size=1000000,
        batch_size=1000,
        print_every=100,
        epochs=1000,
        conv_depth=2,
    ).run()


if __name__ == '__main__':
    main()
    # test()
