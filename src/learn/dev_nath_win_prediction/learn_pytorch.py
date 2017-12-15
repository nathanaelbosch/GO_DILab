from os.path import abspath
import numpy as np
import sqlite3

import torch
from torch.autograd import Variable
import torch.utils.data as Data

from src.learn.BaseLearn import BaseLearn
from src.play.model.Board import EMPTY, BLACK, WHITE


class Learn(BaseLearn):

    def __init__(self):
        super().__init__()
        self.training_size = 100000
        self.data_retrieval_command = '''SELECT games.*, meta.result_text
                                         FROM games, meta
                                         WHERE games.id == meta.id
                                         AND meta.all_moves_imported!=0
                                         ORDER BY RANDOM()
                                         LIMIT ?'''

    def handle_data(self, training_data):
        results_array = training_data[:, -1]
        # ids = training_data[:, 0]
        # colors = training_data[:, 1]
        moves = training_data[:, 2].astype(int)
        boards = training_data[:, 3:-1].astype(np.float64)

        # Moves as categorical data
        moves[moves==-1] = 81

        # Generate symmetries:
        boards, results_array = self.get_symmetries(
            boards, other_data=results_array)

        # Input: Board
        X = np.concatenate(
            ((boards==WHITE)*3 - 1,
             (boards==BLACK)*3 - 1,
             (boards==EMPTY)*3 - 1),
            axis=1)
        X = X / np.sqrt(2)
        print('X.mean():', X.mean())
        print('X.var():', X.var())

        # Output: Result
        results = np.chararray(results_array.shape)
        results[:] = results_array[:]
        # black_wins = results.lower().startswith(b'b')[:, None]
        # white_wins = results.lower().startswith(b'w')[:, None]
        # draws = results.lower().startswith('D')
        # y = np.concatenate((black_wins, white_wins), axis=1)

        black_wins = results.lower().startswith(b'b')
        y = black_wins
        y = y.astype(int)

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def setup_and_compile_model(self):
        pass

    def train(self, model, X, Y):
        model.fit(X, Y, epochs=8, batch_size=10000)

    def get_path_to_self(self):
        return abspath(__file__)

    def run(self):
        # Get data from Database
        cursor = self.db.cursor()
        cursor.execute(self.data_retrieval_command,
                       [self.training_size])
        training_data = np.array(cursor.fetchall())  # this is a gigantic array, has millions of rows

        self.log('working with {} rows'.format(len(training_data)))
        X, y = self.handle_data(training_data)
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y)
        num_classes = 2

        dataset = Data.TensorDataset(data_tensor=X, target_tensor=y)
        data_loader = Data.DataLoader(dataset, batch_size=10, shuffle=True)

        D_in = X.size()[1]
        H1, H2, H3 = 100, 200, 100

        # Pytorch stuff
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.ReLU(),
            # torch.nn.Linear(H1, H2),
            # torch.nn.ReLU(),
            # torch.nn.Linear(H2, H3),
            # torch.nn.ReLU(),
            torch.nn.Linear(H1, num_classes),
            torch.nn.Softmax(dim=1),
        )
        criterion = torch.nn.CrossEntropyLoss()
        # learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(5):
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()

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
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[{:d}, {:5d}]  loss: {:.3f}  accuracy: {:.3f}%'.format(epoch + 1, i + 1, running_loss / 2000, accuracy))
                    running_loss = 0.0
                    correct = 0
                    total = 0
        print('Finished Training')


if __name__ == '__main__':
    Learn().run()
