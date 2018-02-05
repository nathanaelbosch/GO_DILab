import os
import sqlite3
import random
import logging
import argparse

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
from .our_model import ConvNet


###############################################################################
# Logger
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    format='%(levelname)s|%(name)s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info('Logger set up')

###############################################################################
# Parse args
###############################################################################
logger.info('Parse args')
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--no-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--single-gpu', action='store_true',
                    help='Only use a signle gpu')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Epochs to train')
parser.add_argument('-n', '--training-size', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('-d', '--depth', type=int, default=9,
                    help='Number of residual blocks')
parser.add_argument('-f', '--filters', type=int, default=256,
                    help='Number of convolution filters')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.num_gpu = 0 if not args.cuda else (1 if args.single_gpu else torch.cuda.device_count())


###############################################################################
# Directory setup
###############################################################################
logger.info('Set up directories')
DB_PATH = 'data/db.sqlite'
CONV_DIR = 'src/learn/conv'
CURRENT_RUN_DIR = os.path.join(
    CONV_DIR, 'saved', '{}depth_{}filters_{}mtsize'.format(
        args.depth,
        args.filters,
        int(args.training_size/100000)/10))
MODELS_DIR = os.path.join(CURRENT_RUN_DIR, 'nets')
STATISTICS_DIR = os.path.join(CURRENT_RUN_DIR, 'statistics')
if not os.path.exists(CURRENT_RUN_DIR):
    os.makedirs(CURRENT_RUN_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(STATISTICS_DIR):
    os.makedirs(STATISTICS_DIR)
for old_model in os.listdir(MODELS_DIR):
    os.remove(os.path.join(MODELS_DIR, old_model))


###############################################################################
# Connect to db and get data
###############################################################################
logger.info('Connect to database and get data')
data = pd.read_sql_query(
    '''SELECT * FROM games LIMIT ?''', sqlite3.connect(DB_PATH),
    params=[args.training_size])


###############################################################################
# Format data
###############################################################################
logger.info('Format the data')
_bi = 4               # Board column index start
assert data.columns[_bi] == 'loc_0_0_0', 'Board columns bad'
boards = data[data.columns[_bi:(_bi+81)]].as_matrix().reshape(-1, 9, 9)

# Input
colors = data['color'].values
X = minimal_network_input(boards, colors)

# Policy Output
y = policy_output_categorical(data['move'].values)

assert X.shape[0] == y.shape[0], 'Something went wrong with the shapes'
assert X.shape[1] == 2, 'Shouldn\'t this script be minimal?'

logger.debug('X.shape: {}\tX.dtype: {}'.format(X.shape, X.dtype))
logger.debug('y.shape: {}\ty.dtype: {}'.format(y.shape, y.dtype))
logger.debug('Policy Ouput shape: {}'.format(y.shape))

example = random.choice(range(X.shape[0]))
logger.debug('Example input: {}'.format(X[example]))
logger.debug('Example output: {}'.format(y[example]))


###############################################################################
# Train-test split & to pytorch
###############################################################################
logger.info('Train test split')
msk = np.random.rand(X.shape[0]) < 0.9
X, y, X_test, y_test = X[msk], y[msk], X[~msk], y[~msk]
X, y, X_test, y_test = (
    torch.from_numpy(X).float(),
    torch.from_numpy(y),
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test))


###############################################################################
# Create model
###############################################################################
logger.info('Create model')
model = ConvNet(
    X.size(1),
    conv_depth=args.depth,
    n_filters=args.filters)


###############################################################################
# Train
###############################################################################
logger.info('Start training')
policy_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

if args.cuda:
    print('Using cuda')
    if args.num_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.cuda()
    policy_criterion.cuda()

trainset = Data.TensorDataset(data_tensor=X, target_tensor=y)
train_loader = Data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True,
    num_workers=4*args.num_gpu)
testset = Data.TensorDataset(
    data_tensor=X_test, target_tensor=y_test)
val_loader = Data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=True)

results = {
    'policy_loss': [],
    'value_loss': [],
    'train_policy_acc': [],
    'val_policy_acc': [],
    'train_value_acc': [],
    'val_value_acc': [],
}

print('Start training')
for epoch in range(1, args.epochs + 1):
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
        if args.cuda:
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
                      epoch, args.epochs,
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
        if args.cuda:
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
        os.path.join(MODELS_DIR, filename))

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
        filepath = os.path.join(STATISTICS_DIR, filename)
        data_frame.to_csv(filepath, index_label='epoch')

logger.info('Finished Training')
