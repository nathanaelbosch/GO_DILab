import os
import sqlite3
import random
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    import torch
    from torch.autograd import Variable
    import torch.utils.data as Data
except Exception:
    pass

from .utils import (encode_board, value_output, policy_output_categorical,
                    network_input)
from .model_zero import ConvNet


logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    format='%(levelname)s|%(name)s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            FROM games
            LIMIT ?'''
        self.db = sqlite3.connect(DB_PATH)

        self.epochs = kwargs.get('epochs', 5)
        self.batch_size = kwargs.get('batch_size', 100)
        self.conv_depth = kwargs.get('conv_depth', 5)
        self.n_filters = kwargs.get('n_filters', 64)
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

        # Get boards
        _bi = 4               # Board column index start
        assert data.columns[_bi] == 'loc_0_0_0'
        boards = data[data.columns[_bi:(_bi+81)]].as_matrix().reshape(-1, 9, 9)

        # Input
        colors = data['color'].values
        X = network_input(boards, colors)

        # Policy Output
        policy_y = policy_output_categorical(data['move'].values)

        # Value Output
        value_y = value_output(data['result'], data['color'].values)

        y = np.column_stack((policy_y, value_y))

        assert X.shape[0] == y.shape[0], 'Something went wrong with the shapes'

        logger.debug('X.shape:', X.shape, '\tX.dtype:', X.dtype)
        logger.debug('y.shape:', y.shape, '\ty.dtype:', y.dtype)
        logger.debug('Policy Ouput shape:', policy_y.shape)
        logger.debug('Value Ouput shape:', value_y.shape)

        example = random.choice(range(X.shape[0]))
        logger.debug('Example input:', X[example])
        logger.debug('Example output:', y[example])

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
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = torch.nn.DataParallel(model)
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

                predicted_result = torch.sign(value_output.data).view(-1)
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

                predicted_result = torch.sign(value_output.data).view(-1)
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
        logger.info('[0] Get data from db')
        data = self.get_data()
        logger.info('[1] Format data')
        X, y = self.format_data(data)

        if self.test:
            X_train, y_train, X_test, y_test = self.train_test_split(X, y)
            X_train, y_train, X_test, y_test = self.to_pytorch(
                X_train, y_train, X_test, y_test)
            self.X_train, self.y_train, self.X_test, self.y_test = (
                X_train, y_train, X_test, y_test)
        else:
            self.X_train, self.y_train = self.to_pytorch(X, y)

        in_channels = self.X_train.size(1)
        logger.info('[2] Create model')
        self.model = ConvNet(
            in_channels,
            conv_depth=self.conv_depth,
            n_filters=self.n_filters)

        logger.info('[3] Start the training')
        self.train_model()


def test():
    Learn(
        training_size=100,
        batch_size=10,
        epochs=3,
        conv_depth=19,
        # no_cuda=True,
    ).run()


def overfit():
    Learn(
        training_size=100,
        batch_size=10,
        epochs=1000,
        conv_depth=9,
        no_cuda=True,
        # test=False,
        # symmetries=True,
    ).run()


"""
Notes on training size:
10m: 15GB
nolimit: 42
"""


def main():
    SETUPS = {
        'google_cloud': {
            'training_size': 1,
            'batch_size': 1
        },
        'dgx1': {
            'training_size': 1000000,
            'batch_size': 8*1000
        },
        'titanx_16ram_kwargs': {
            'training_size': 2000000,
            # Batch size depends on model size: BIG:1000, normal:10000
            'batch_size': 1000,
        },
        'nv1050_8ram_kwargs': {
            'training_size': 1000000,
            'batch_size': 2000,
        }
    }

    Learn(
        **SETUPS['dgx1'],
        # **nv1050_8ram_kwargs,
        epochs=5000,
        conv_depth=19,
        n_filters=256,
    ).run()


if __name__ == '__main__':
    main()
    # test()
    # overfit()
