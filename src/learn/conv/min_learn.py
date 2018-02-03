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

from .utils import policy_output_categorical, minimal_network_input
from .min_model import ConvNet


logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    format='%(levelname)s|%(name)s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DB_PATH = 'data/db.sqlite'
CONV_DIR = 'src/learn/conv'


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

        self.dir_setup()

    def dir_setup(self):
        self.CURRENT_DIR = os.path.join(
            CONV_DIR, '{}depth_{}filters_{}mtsize'.format(
                self.conv_depth,
                self.n_filters,
                int(self.training_size/100000)/10))
        self.MODELS_DIR = os.path.join(self.CURRENT_DIR, 'nets')
        self.STATISTICS_DIR = os.path.join(self.CURRENT_DIR, 'statistics')
        if not os.path.exists(self.CURRENT_DIR):
            os.makedirs(self.CURRENT_DIR)
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)
        if not os.path.exists(self.STATISTICS_DIR):
            os.makedirs(self.STATISTICS_DIR)
        for old_model in os.listdir(self.MODELS_DIR):
            os.remove(os.path.join(self.MODELS_DIR, old_model))

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
        X = minimal_network_input(boards, colors)

        # Policy Output
        y = policy_output_categorical(data['move'].values)

        assert X.shape[0] == y.shape[0], 'Something went wrong with the shapes'
        assert X.shape[1] == 2, 'Shouldn\'t this script be minimal?'

        logger.debug('X.shape:', X.shape, '\tX.dtype:', X.dtype)
        logger.debug('y.shape:', y.shape, '\ty.dtype:', y.dtype)
        logger.debug('Policy Ouput shape:', y.shape)

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
        optimizer = torch.optim.Adam(model.parameters())

        if self.use_cuda:
            print('Using cuda')
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = torch.nn.DataParallel(model)
            model.cuda()
            policy_criterion.cuda()

        trainset = Data.TensorDataset(data_tensor=X, target_tensor=y)
        train_loader = Data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True,
            num_workers=4*torch.cuda.device_count() if self.use_cuda else 4)
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
                'policy_loss': 0, 'policy_correct': 0, 'policy_accuracy': 0}

            model.train()
            for data, targets in train_bar:
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size

                data = Variable(data)
                targets = Variable(targets)
                if self.use_cuda:
                    data = data.cuda()
                    targets = targets.cuda()

                model.zero_grad()
                policy_output = model(data)
                policy_loss = policy_criterion(policy_output, targets)
                policy_loss.backward()
                optimizer.step()

                _, predicted_move = torch.max(policy_output.data, 1)
                running_results['policy_loss'] += (
                    policy_loss.data[0] * batch_size)
                running_results['policy_correct'] += (
                    predicted_move == targets.data).sum()
                running_results['policy_accuracy'] = (
                    100 * running_results['policy_correct'] /
                    running_results['batch_sizes'])

                total_seen = running_results['batch_sizes']
                train_bar.set_description(
                    desc=('[{:d}/{:d}] Policy-Loss: {:.4f} ' +
                          'Policy-Accuracy: {:.2f}% ').format(
                              epoch, self.epochs,
                              running_results['policy_loss'] / total_seen,
                              running_results['policy_accuracy']))

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

                policy_output = model(data)

                _, predicted_move = torch.max(policy_output.data, 1)
                val_results['policy_correct'] += (
                    predicted_move == target.data).sum()
                val_results['policy_accuracy'] = (
                    100 * val_results['policy_correct'] /
                    val_results['batch_sizes'])

            print('[Validation] Policy-accuracy: {:.2f}'
                  .format(val_results['policy_accuracy']))

            # Save model parameters
            filename = 'epoch{}.pth'.format(epoch)
            torch.save(
                model.state_dict(),
                os.path.join(self.MODELS_DIR, filename))

            # Save statistics
            total_seen = running_results['batch_sizes']
            results['policy_loss'].append(
                running_results['policy_loss'] / total_seen)
            results['train_policy_acc'].append(
                running_results['policy_accuracy'])
            results['val_policy_acc'].append(
                val_results['policy_accuracy'])
            if epoch % 1 == 0 and epoch != 0:
                data_frame = pd.DataFrame(
                    data={'train_loss': results['policy_loss'],
                          'val_acc': results['val_policy_acc'],
                          'train_acc': results['train_policy_acc'],
                          },
                    index=range(1, epoch + 1))
                filename = 'train_results.csv'
                filepath = os.path.join(self.STATISTICS_DIR, filename)
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
        conv_depth=1,
        n_filters=3,
        no_cuda=True,
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
            'training_size': 20000000,
            'batch_size': 8*1000
        },
        'titanx_16ram': {
            'training_size': 1700000,
            # Batch size depends on model size:
            # BIG:1000
            # 9depth64filter:8000
            # 9depth32filter:8000
            'batch_size': 8000,
        },
        'nv1050_8ram_kwargs': {
            'training_size': 1000000,
            'batch_size': 2000,
        }
    }

    Learn(
        **SETUPS['titanx_16ram'],
        # **nv1050_8ram_kwargs,
        epochs=5000,
        conv_depth=9,
        n_filters=256,
    ).run()


if __name__ == '__main__':
    main()
    # test()
    # overfit()
