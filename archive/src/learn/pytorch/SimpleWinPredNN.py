"""SimpleWinPredNN"""
import torch
import torch.nn as nn


class SimpleWinPredNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super(SimpleWinPredNN, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

    def save_state_dict(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)
