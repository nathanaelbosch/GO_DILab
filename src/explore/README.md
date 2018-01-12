# Reinforcement Learning Approach using OpenAI gym

## Setup
Install OpenAI gym according to their [github page](https://github.com/openai/gym), or directly from PyPI:
```
pip install gym
pip install gym[board_game]
```

## Examples
- Random playing: `src/explore/.open_ai_examples/go.py`
- [Deep Q Network example using pytorch](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) on the cartpole task: `src/explore/.open_ai_examples/reinforcement_q_learning.py`

## What I did
I used the pytorch example and tried to make it run on GO while doing only minimal changes to the script. The good: It runs! The bad: It's bad.
