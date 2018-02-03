from src.play.model.Game import Game
from src.play.model.Game import WHITE, BLACK, EMPTY
from src.play.model.Move import Move
from src.learn.RL_Atari.train_atari import init_game
from src.learn.RL_Atari.train_atari import board2input
import random
import numpy as np
import copy
import time

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

def main():
    model = keras.models.load_model('src/learn/RL_Atari/test_model_1.h5')
    game = Game()
    col_coord, row_coord = 1,6
    game = init_game(game,col_coord,row_coord)
    print('new game')
    print(game)
    k = 0
    #print(model.predict(board2input(game,'b'),batch_size=1))
    #time.sleep(40)
    while k < 4:
        qval = model.predict(board2input(game,'b'),batch_size=1)
        #print(qval)
        #time.sleep(100)
        temp_qval = copy.copy(qval)
        move = np.argmax(qval)
        #print(move)
        move = Move.from_flat_idx(move)
        location = move.to_matrix_location()
        while game.board[location] != EMPTY:
            temp_qval[0][np.argmax(temp_qval)] = -100  # arbit low value. To get to second max value.
            move = np.argmax(temp_qval)
            move = Move.from_flat_idx(move)
            location = move.to_matrix_location()
        game.play(move, 'b')
        print(game)
        k = k + 1



    # print(game)



if __name__ == '__main__':
    main()

