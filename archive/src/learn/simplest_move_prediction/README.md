# Simplest Move prediction from board state, learned from historic data

Really the simplest situation I can think of. Try to recreate the moves present in 
the training data, by training a NeuralNetwork on it.

## Description of the Neural Network
**Input**: The current board - 81-dim vector of values in {0,1,-1}
**Output**: (81+1)-dim vector, using softmax to decide on a move (+1 as passing is also a move)

