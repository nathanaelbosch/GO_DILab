import os
import sqlite3
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.utils.data as Data

import src.learn.bots.utils as utils
from .model_zero import ConvNet


DB_PATH = 'data/db.sqlite'
CONV_DIR = 'src/learn/conv'
MODELS_DIR = os.path.join(CONV_DIR, 'nets')
STATISTICS_DIR = os.path.join(CONV_DIR, 'statistics')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(STATISTICS_DIR):
    os.makedirs(STATISTICS_DIR)
for old_model in os.listdir(MODELS_DIR):
    os.remove(os.path.join(MODELS_DIR, old_model))


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
        self.conv_depth = kwargs.get('conv_depth', 5)
        no_cuda = kwargs.get('no_cuda', False)
        self.use_cuda = (
            False if no_cuda or not torch.cuda.is_available() else True)
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

        # Input
        X = utils.encode_board(boards, data['color'])
        X = X.reshape(-1, 3, 9, 9).astype(float)

        # Policy Output
        passes = data['move'] == -1
        policy_y = data['move']
        policy_y[passes] = 81
        policy_y = policy_y.values

        # Value Output
        value_y = utils.value_output(data['result'], data['color'])
        value_y = value_y[:, 0]

        y = np.concatenate(
            (policy_y.reshape(-1, 1), value_y.reshape(-1, 1)), axis=1)
        # X = X[~passes]
        # y = y[~passes]
        assert X.shape[0] == y.shape[0], 'Something went wrong with the shapes'
        assert (X.sum(axis=0).sum(axis=0) == X.shape[0]).all()
        print('X.shape:', X.shape, 'X.dtype:', X.dtype)
        print('y.shape:', y.shape, 'y.dtype:', y.dtype)
        print('Policy Ouput shape:', policy_y.shape)
        print('Value Ouput shape:', value_y.shape)

        # example = random.choice(range(X.shape[0]))
        # print('Example input:', X[example])
        # print('Example output:', y[example])

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
        policy_criterion = torch.nn.CrossEntropyLoss()
        value_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        if self.use_cuda:
            print('Using cuda')
            model.cuda()
            policy_criterion.cuda()
            value_criterion.cuda()

        trainset = Data.TensorDataset(data_tensor=X, target_tensor=y)
        train_loader = Data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True)
        testset = Data.TensorDataset(
            data_tensor=self.X_test, target_tensor=self.y_test)
        val_loader = Data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=True)

        results = {
            'policy_loss': [],
            'value_loss': [],
            'train_policy_acc': [],
            'val_policy_acc': [],
            'train_value_acc': [],
            'val_value_acc': [],
        }

        print('Start training')
        for epoch in range(1, self.epochs + 1):
            train_bar = tqdm(train_loader)
            running_results = {
                'batch_sizes': 0,
                'policy_loss': 0, 'policy_correct': 0, 'policy_accuracy': 0,
                'value_loss': 0, 'value_correct': 0, 'value_accuracy': 0}

            model.train()
            for data, targets in train_bar:
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size

                data = Variable(data)
                targets = Variable(targets)
                if self.use_cuda:
                    data = data.cuda()
                    targets = targets.cuda()
                policy_target, value_target = targets[:, 0], targets[:, 1]

                model.zero_grad()
                policy_output, value_output = model(data)
                policy_loss = policy_criterion(policy_output, policy_target)
                value_loss = value_criterion(
                    value_output, value_target.float())
                loss = policy_loss + 0.01 * value_loss
                loss.backward()
                optimizer.step()

                _, predicted_move = torch.max(policy_output.data, 1)
                running_results['policy_loss'] += (
                    policy_loss.data[0] * batch_size)
                running_results['policy_correct'] += (
                    predicted_move == policy_target.data).sum()
                running_results['policy_accuracy'] = (
                    100 * running_results['policy_correct'] /
                    running_results['batch_sizes'])

                predicted_result = torch.round(value_output.data).view(-1)
                running_results['value_loss'] += (
                    value_loss.data[0] * batch_size)
                running_results['value_correct'] += (
                    predicted_result == value_target.data.float()).sum()
                running_results['value_accuracy'] = (
                    100 * running_results['value_correct'] /
                    running_results['batch_sizes'])

                total_seen = running_results['batch_sizes']
                train_bar.set_description(
                    desc=('[{:d}/{:d}] Policy-Loss: {:.4f} ' +
                          'Value-Loss: {:.4f} Policy-Accuracy: {:.2f}% ' +
                          'Value-Accuracy: {:.2f}%').format(
                              epoch, self.epochs,
                              running_results['policy_loss'] / total_seen,
                              running_results['value_loss'] / total_seen,
                              running_results['policy_accuracy'],
                              running_results['value_accuracy']))

            model.eval()
            # val_bar = tqdm(val_loader)
            val_results = {
                'policy_correct': 0, 'policy_accuracy': 0, 'batch_sizes': 0,
                'value_correct': 0, 'value_accuracy': 0}
            for data, target in val_loader:
                batch_size = data.size(0)
                val_results['batch_sizes'] += batch_size

                data = Variable(data, volatile=True)
                target = Variable(target, volatile=True)
                if self.use_cuda:
                    data = data.cuda()
                    target = target.cuda()
                policy_target, value_target = target[:, 0], target[:, 1]

                policy_output, value_output = model(data)

                _, predicted_move = torch.max(policy_output.data, 1)
                val_results['policy_correct'] += (
                    predicted_move == policy_target.data).sum()
                val_results['policy_accuracy'] = (
                    100 * val_results['policy_correct'] /
                    val_results['batch_sizes'])

                predicted_result = torch.round(value_output.data).view(-1)
                val_results['value_correct'] += (
                    predicted_result == value_target.data.float()).sum()
                val_results['value_accuracy'] = (
                    100 * val_results['value_correct'] /
                    val_results['batch_sizes'])

            print('[Validation] Policy-accuracy: {:.2f} Value-accuracy {:.2f}'
                  .format(val_results['policy_accuracy'],
                          val_results['value_accuracy']))

            # Save model parameters
            torch.save(
                model.state_dict(),
                os.path.join(MODELS_DIR, 'convnet_epoch_{}.pth'.format(epoch)))

            # Save statistics
            total_seen = running_results['batch_sizes']
            results['policy_loss'].append(
                running_results['policy_loss'] / total_seen)
            results['value_loss'].append(
                running_results['value_loss'] / total_seen)
            results['train_policy_acc'].append(
                running_results['policy_accuracy'])
            results['train_value_acc'].append(
                running_results['value_accuracy'])
            results['val_policy_acc'].append(
                val_results['policy_accuracy'])
            results['val_value_acc'].append(
                val_results['value_accuracy'])
            if epoch % 1 == 0 and epoch != 0:
                data_frame = pd.DataFrame(
                    data={'policy_loss': results['policy_loss'],
                          'value_loss': results['value_loss'],
                          'val_policy_acc': results['val_policy_acc'],
                          'val_value_acc': results['val_value_acc'],
                          'train_policy_acc': results['train_policy_acc'],
                          'train_value_acc': results['train_value_acc'],
                          },
                    index=range(1, epoch + 1))
                filename = 'train_results.csv'
                filepath = os.path.join(STATISTICS_DIR, filename)
                data_frame.to_csv(filepath, index_label='epoch')

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

        self.model = ConvNet(
            self.X_train.size(),
            self.y_train.size(),
            self.batch_size)

        self.train_model()


def test():
    _l = Learn(
        training_size=10,
        batch_size=1,
        epochs=3,
        conv_depth=2,
        # no_cuda=True,
    )
    _l.data_retrieval_command = '''
        SELECT *
        FROM elo_ordered_games
        LIMIT ?'''
    _l.run()


def overfit():
    _l = Learn(
        training_size=100,
        batch_size=10,
        epochs=1000,
        conv_depth=9,
        no_cuda=True,
        # test=False,
        # symmetries=True,
    )
    _l.data_retrieval_command = '''
        SELECT *
        FROM elo_ordered_games
        LIMIT ?'''
    _l.run()


def main():
    # 1m training size for 8GB machine
    # 2.5m for 16GB
    Learn(
        training_size=2500000,
        batch_size=1000,
        epochs=5000,
        conv_depth=19,
        symmetries=True,
    ).run()


if __name__ == '__main__':
    main()
    # test()
    # overfit()
