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


    epochs = 50000
    gamma = 0.975
    epsilon = 1
    batchSize = 50
    buffer = 100
    replay = []
    h = 0
    for i in range(epochs):
        col_coord, row_coord = random.randint(0, 8), random.randint(0, 8)
        #print(col_coord,row_coord)
        game = Game()
        game = init_game(game,col_coord,row_coord)
        status = 1
        reward = -1 # by default at game start
        # game in progress
        while (status==1):
            qval = model.predict(board2input(game,'b'), batch_size=1)
            if reward == -1:
                if (random.random() < epsilon):
                    valid_moves = game.get_playable_locations(BLACK)
                    move = random.choice(valid_moves)
                    while move.is_pass == True:
                        move = random.choice(valid_moves)
                        if len(valid_moves) == 0:
                            print('end it')
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
                reward = 50
            else:
                reward = -1

            # experience replay storage
            if len(replay) < buffer:
                replay.append((board2input(game,'b'),move,reward,board2input(new_game,'b')))
            else:
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (board2input(game,'b'),move,reward,board2input(new_game,'b'))
                minibatch = random.sample(replay,batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    (m_game,m_move,m_reward,m_new_game) = memory
                    oldqval = model.predict(m_game,batch_size=1)
                    maxq = oldqval[0][m_move]
                    y = np.zeros(81)
                    y[:] = oldqval
                    if m_reward == 50:
                        update = m_reward
                    else:
                        update = m_reward + gamma * maxq
                    y[m_move] = update
                    X_train.append(m_game)
                    y_train.append(y)
                X_train = np.stack(X_train)
                y_train = np.stack(y_train)
                #print('ytrain: ', y_train[0])
                model.fit(X_train,y_train,batch_size=batchSize,epochs=1,verbose=0)
            game = copy.copy(new_game)
            if reward == 50:
                status = 0
        print ('game ' + str(i) + ' ends here')
        #print(game)
        #temp_move = Move.from_flat_idx(move)
        #print(temp_move)
        #print(model.predict(board2input(game,'b'),batch_size=1))
        #input()
        if epsilon > 0.1:
            epsilon -= (1 / epochs)
            #print ('epsilon : ' + str(epsilon))
        if i % 5000 == 0 and i > 0:
            name = 'src/learn/RL_Atari/hard_atari_' + str(i) + '.h5'
            model.save(name)

    model.save('src/learn/RL_Atari/test_model_final.h5')
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

