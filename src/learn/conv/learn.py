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
            -- ORDER BY RANDOM()
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
        assert X.shape[0] == y.shape[0], 'Something went wrong with the shapes'
        print('X.shape:', X.shape, 'X.dtype:', X.dtype)
        print('y.shape:', y.shape, 'y.dtype:', y.dtype)

        X = X[~passes]
        y = y[~passes]

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y)
        print('X.size:', X.size(), 'X.dtype:', X.type())
        print('y.size:', y.size(), 'y.dtype:', y.type())

        return X, y

    def train_model(self, X, y):
        model = ConvNet(X.size(), y.size())
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
        print('Finished Training')

    def run(self):
        data = self.get_data()
        X, y = self.format_data(data)

        self.train_model(X, y)


def main():
    Learn(
        training_size=100000,
        batch_size=1000,
        print_every=10,
        epochs=50,
        conv_depth=2,
    ).run()


if __name__ == '__main__':
    main()
