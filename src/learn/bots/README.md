# Compare bot ideas

The multiple different approaches so far differ in three points:
1. NN Output and subsequently Bot *logic*
2. Input data given to the NN
3. Network design

They also differed on which data they were trained, how they were implemented, how long they have been trained etc, but those difference are problematic as they lead to unfair comparisons.

I therefore want to summarize the approaches, find out which variants should be compared and tested, implement them, and train them on the same pre-defined choice and amount of data.

### 1. Output
It basically came down to two main ideas on the bot logic:
- Predicting the outcome of the game, and then chosing the best option:  
  $\mathcal{I} \to [0, 1]^2$
- Directly *learning* to outpt a good move:  
  $\mathcal{I} \to [0, 1]^{9*9 + 1}$
I will refer to those two approaches as **Value Network** and **Policy Network**.


### 2. Input
I propose three different types of input:
1. Board as input, but *badly* encoded: $\mathcal{I}=\{-1, 0, 1\}^81$, normalized according to the data
2. Board as input, but *better* encoded: $\mathcal{I}=\{-1/\sqrt{2}, 1/\sqrt{2}\}^{3*81}$

### 3. Network design
tbd.
I'd just try some architectures and settle on one or two versions that perform reasonably well. Maybe do some more testing to find valid arguments for a specific architecture.

## My idea on how to proceed here
First we need to settle on a good part of the dataset that will be used for this comparison. Then, for all three input types I want to train both a policy and a value network on this data. Lastly I would evaluate all the resulting six bots by playing n games against the random bot, and preferrably also against gnugo (or something similar).

## Naming
11: Value bot with naive board encoding into a 81-vector  
12: Policy bot with naive board encoding into a 81-vector  
21: Value bot with the advanced board encoding into a (3*81)-vector  
22: Policy bot with the advanced board encoding into a (3*81)-vector  

## ToDo
- [x] Research on some papers again to settle on a reasonable choice for the liberty encoding
- [x] Change encodings from a "black-white" to a "player-opponent" point of view
- [ ] Understand the `LibertyNNBot` to get the liberties
- [ ] Add normalizing to the naive board encoding variants
- [ ] Maybe add some variants for the models to compare - Might be better to do after comparing those 6 versions to then compare models
- [x] Implement a reasonable SQL query to get a good choice of training data
- [x] Implement the play-logic for both PolicyBot and ValueBot

