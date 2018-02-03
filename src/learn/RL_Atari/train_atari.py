from src.play.model.Game import Game
from src.play.model.Game import WHITE, BLACK, EMPTY
from src.play.model.Move import Move
import random
import numpy as np
import copy
import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

def board2input(game,player):
    if player == 'b':
        me = BLACK
        other = WHITE
    else:
        me = WHITE
        other = BLACK
    b = game.board
    my_board = (b == me)*1
    other_board = (b == other)*1
    #empty_board = b == EMPTY
    empty_board = (np.matrix([[1] * 9] * 9)) - my_board - other_board
    my_board = my_board.reshape(my_board.shape[0]*my_board.shape[1],)
    #my_board = my_board.flatten()
    other_board = other_board.reshape(other_board.shape[0] * other_board.shape[1],)
    empty_board = empty_board.reshape(empty_board.shape[0] * empty_board.shape[1],)
    vect = np.concatenate((my_board,other_board,empty_board),1)
    return vect

def check_dead_group(game,col_coord,row_coord):
    b = game.board
    total_neighbors = []
    loc = Move(col=col_coord,row = row_coord).to_matrix_location()
    total_neighbors = b.get_adjacent_coords(loc)
    for n in total_neighbors:
        if b[n] == EMPTY:
            return False
    return True


def init_game(game,col_coord,row_coord):
    move = Move(col=col_coord, row=row_coord)
    game.play(move,'w')
    return game

def main():
    model = Sequential()
    model.add(Dense(units = 200,kernel_initializer = 'uniform',activation='relu',input_shape=(243,)))
    model.add(Dense(units = 400,kernel_initializer = 'uniform',activation='relu'))
    model.add(Dense(units = 200,kernel_initializer = 'uniform',activation='relu'))
    model.add(Dense(units = 81,kernel_initializer = 'uniform',activation='linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    col_coord, row_coord = 1,6#random.randint(0, 8), random.randint(0, 8)
    epochs = 10
    gamma = 0.9
    epsilon = 1
    for i in range(epochs):

        game = Game()
        game = init_game(game,col_coord,row_coord)
        status = 1
        # game in progress
        while (status==1):
            qval = model.predict(board2input(game,'b'), batch_size=1)
            if (random.random() < epsilon):
                valid_moves = game.get_playable_locations('b')
                move = random.choice(valid_moves)
                while move.is_pass == True:
                    move = random.choice(valid_moves)
                new_game = copy.deepcopy(game)
                new_game.play(move, 'b')
                move = move.to_flat_idx()
            else:
                temp_qval = copy.copy(qval)
                move = (np.argmax(temp_qval))
                move = Move.from_flat_idx(move)
                new_game = copy.deepcopy(game)
                location = move.to_matrix_location()
                while new_game.board[location] != EMPTY:
                    temp_qval[0][np.argmax(temp_qval)] = -100 # arbit low value. To get to second max value.
                    move = np.argmax(temp_qval)
                    move = Move.from_flat_idx(move)
                    location = move.to_matrix_location()
                new_game.play(move,'b')
                move = move.to_flat_idx()

            if check_dead_group(new_game,col_coord,row_coord)==True:
                reward = 10
                status = 0
            else:
                reward = -1

            # get maxQ from new state
            newQ = model.predict(board2input(game,'b'),batch_size=1)
            maxQ = newQ[0][move]
            # update, reward : update = reward if reward = 100, else = reward + gamma*maxQ
            if reward == -1:  # non-terminal state
                update = (reward + (gamma * maxQ))
            else:  # terminal state
                update = reward
            # set y = qval, and y[action] = update => assigning reward value for action.
            y = np.zeros((1,81))
            y[:] = qval[:]
            y[0][move] = update
            # fit the model according to present shape and y
            model.fit(board2input(game,'b'),y, batch_size=1, nb_epoch=1, verbose=0)
            game = copy.copy(new_game)
        print ('game ' + str(i) + ' ends here')
        if epsilon > 0.1:
            epsilon -= (1 / epochs)
            #print ('epsilon : ' + str(epsilon))

    model.save('test_model_1.h5')
    #print (col_coord,row_coord)
    # game = Game()
    # game = init_game(game,col_coord,row_coord)
    # print('new game')
    # print(game)
    # for i in range(3):
    #     qval = model.predict(board2input(game,'b'), batch_size=1)
    #     print (qval)
    #     action = (np.argmax(qval))
    #     invalid_moves = game.get_invalid_locations('b')
    #     move = Move.from_flat_idx(action)
    #     if move in invalid_moves:
    #         qval[np.nonzero(qval==action)[0][0]] = -100
    #         action = (np.argmax(qval))
    #         move = Move.from_flat_idx(action)
    #     game.play(move,'b')
    #     print('move : ' + str(i))
    #     print(game)





if __name__ == '__main__':
    main()

