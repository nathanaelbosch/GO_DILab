import numpy as np
from src import Utils
from src.play.model.Board import WHITE, BLACK, EMPTY
import os
from scipy import ndimage
from os.path import abspath, dirname
from src.play.model.Move import Move


class LibertyNNBot:

    def __init__(self):
        project_dir = dirname(dirname(dirname(dirname(
            abspath(__file__)))))
        Utils.set_keras_backend("theano")
        import keras
        model_path = os.path.join(
            project_dir, 'src/learn/dev_kar/model.h5')
        self.model = keras.models.load_model(model_path)
        mean_var_path = os.path.join(project_dir, 'src/learn/dev_kar/mean_var.txt')
        with open(mean_var_path, 'r') as f:
            lines = f.readlines()
        self.mean = float(lines[0])
        self.std = float(lines[1])

    def board_to_input(self, color, board):
        b = board.astype(np.float64)

        # print(b.dtype)
        # print(type(self.mean))
        # b -= self.mean
        # b /= self.std
        if color == 'b':
            me = BLACK
            other = WHITE
        else:
            me = WHITE
            other = BLACK

        my_board = (b == me) * 1
        other_board = (b == other) * 1
        empty_board = (np.matrix([[1.0] * 9] * 9)) - my_board - other_board
        empty_board = empty_board / np.count_nonzero(empty_board)
        my_board_vals = np.matrix([[0.0] * 9] * 9)
        other_board_vals = np.matrix([[0.0] * 9] * 9)

        label_mine, mine_labels = ndimage.label(my_board)
        label_other, other_labels = ndimage.label(other_board)

        for label in range(1, mine_labels + 1):
            my_board_label = (label_mine == label) * 1
            dilated = ndimage.binary_dilation(my_board_label)  # dilates a group
            dilated = ((dilated - other_board - my_board_label) == 1)  # gets the net increase of each group
            L = np.count_nonzero(dilated)  # L = Total number of liberties of group
            stone_list = list(zip(np.where(my_board_label)[0], np.where(my_board_label)[1]))  # present group location
            for location in stone_list:
                stone_dilated = np.matrix([[0] * 9] * 9)
                stone_dilated[location] = 1
                stone_dilated = ndimage.binary_dilation(stone_dilated)
                stone_liberty = (stone_dilated - other_board - my_board_label) == 1
                sL = np.count_nonzero(stone_liberty)  # liberty per stone
                if L == 0:
                    break
                my_board_vals[location] = sL / L

        for label in range(1, other_labels + 1):
            other_board_label = (label_other == label) * 1
            dilated = ndimage.binary_dilation(other_board_label)
            dilated = ((dilated - other_board_label - my_board) == 1)
            L = np.count_nonzero(dilated)
            stone_list = list(zip(np.where(other_board_label)[0], np.where(other_board_label)[1]))
            for location in stone_list:
                stone_dilated = np.matrix([[0] * 9] * 9)
                stone_dilated[location] = 1
                stone_dilated = ndimage.binary_dilation(stone_dilated)
                stone_liberty = (stone_dilated - other_board - my_board) == 1
                sL = np.count_nonzero(stone_liberty)
                if L == 0:
                    break
                other_board_vals[location] = sL / L

        # print (my_board_vals)
        # print('helloooooo')
        # print(other_board_vals)

        my_board_vect = my_board_vals.reshape(
            1, my_board_vals.shape[0] * my_board_vals.shape[1])
        other_board_vect = other_board_vals.reshape(
            1, other_board_vals.shape[0] * other_board_vals.shape[1])
        empty_board = empty_board.reshape(
            1, empty_board.shape[0] * empty_board.shape[1])

        a = np.append([my_board_vect, other_board_vect, empty_board],[])
        a = np.reshape(a,(1,a.shape[0]))
        return a

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def genmove(self, color, game) -> Move:
        # We're still interested in the playable locations
        playable_locations = game.get_playable_locations(color)

        # Format the board and make predictions
        inp = self.board_to_input(color, game.board)
        pred_moves = self.model.predict(inp)
        pred_moves = pred_moves.reshape(9, 9)

        # print(pred_moves)
        # print(playable_locations)
        dummy_value = -10
        potential_moves = np.array([[dummy_value]*9]*9, dtype=float)
        for move in playable_locations:
            # print(move)
            if move.is_pass:
                continue
            loc = move.to_matrix_location()
            potential_moves[loc[0]][loc[1]] = pred_moves[loc[0]][loc[1]]

        potential_moves = self.softmax(potential_moves)

        row, col = np.unravel_index(
            potential_moves.argmax(),
            potential_moves.shape)

        move = Move(col=col, row=row)
        # if game.board[col,row] != 0:
        #     move = Move(is_pass = True)
        #     return move

        if potential_moves[move.to_matrix_location()] == dummy_value:
            move = Move(is_pass=True)

        return move
