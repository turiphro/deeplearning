Today we discussed two papers from Google DeepMind, both combining deep neural
networks and reinforcement learning.

Reinforcement learning is a paradigm whereby the computer tries to learn the
best actions to take in a specific environment by interacting with this
environment. It learns a _policy_ that maps _states_ to _actions_, by both
exploring (random actions) and exploiting (taking best action from policy so
far), and observing occasional rewards, positive or negative. Those rewards
will 'spread out' to neighbouring states when the policy learns the expected
future reward for every state. RL makes a lot of sense in situations where
interaction with the environment is possible, since the algorithm can steer
towards quick and more informed learning by choosing actions.

In reality, most problems have an enormous state space and cannot be learned
directly. Neural networks generalise well and can therefore be used to learn
the policy in high-dimensional spaces. However, the usual supervised training
needs some workarounds in **"Deep Reinforcement Learning"**, since rewards
are delayed, sparse and noise. Furthermore, the observed state sequences are
highly correlated and the interaction changes the state distribution _while
learning_.

**Atari games**

Learning to play games with reinforcement learning makes a lot of sense;
however, until now the games were super simplistic, or relied on hand-crafted
features. Not with DeepMind: the deep neural network used for learning the
policy takes raw pixels as input and has output neurons for all possible
actions (note that this is a trick to evaluate all actions at once, instead
of using actions as inputs with just one output). To make the state Markovian
(ish) we take the last four frames as inputs. All games use the same network
of two convolutional layers plus one fully connected, so no game-specific
information is used anywhere! All Atari games have intermediate scores, and
thus the RL takes the _change in score_ as the reward. Learning consists of a
loop alternating between:

1. RL: take action, observe (state, action, new state, reward). Save in
   **replay memory**. In theory one can also simply observe humans play.
2. SL: sample mini-batch from replay memory, and train NN with the change
   in `Q(s,a)` as cost. Sampling instead of training on each example breaks
   the sequence correlation and smooths the learning.

This method dwarfs the competition on (almost) all games, and even humans in
two thirds of the Atari games.

**AlphaGo**

The Go board game has been seen as the next 'holy grail' in AI after chess,
and DeepMind solved it. Contrary to the Atari games, in Go there is no
immediate score, so we can only learn from the final result. As usual with
board games, AlphaGo does a tree search while playing to evaluate possible
moves. The tree, however, is massive, so we can't really search it and we'll
need heuristics to go with a Monte Carlo Tree Search. We'll train:

1. SL: **policy network**: a CNN of 13 layers, trained on a dataset of 30M
   Go games played by experts, to learn to predict the best actions given a
   board. We also learn a shallow  **rollout policy** that is 1000x quicker to
   evaluate on data (we'll need it later).
2. RL: improve the large **policy network** by self-play against earlier
   versions of the policy network (to prevent overfitting to current policy).
   As said, only the final reward (+1 or -1) is known, and is used to update
   the `Q(s,a)` values for all moves after a full game is played.
3. SL: we would also like to train another deep network (**value network**)
   to evaluate win chances given specific boards. It outputs a single value
   and is trained on 30M self-played games with only one random move selected
   from each game (to prevent high correlation, thus generalise better).

During game play these elements are combined in a tree search, that repeatedly
expands the three of board moves one leaf further in a partial game that looks
promising (high value in **policy network**, high board evaluation), and
updates the board evaluation of the new leaf by averaging the **value network**
output for the board + the final result of doing one full game (rollout) using
the fast **rollout policy**. This might seem hacky, and it is, but analysis
shows both elements contribute to the win chances. The policy network output
for nodes in the Monte Carlo is weighted down when visited, in order to promote
exploration. When "time is up", the action is taken that has been _visited
most_.

Alpha Go again dwarfs all other Go programs out there and won 4-1 from the
(poor, human) world champion Go.


**Deep Dream**
We played with Googles Deep Dream implementation on github and created crazy
versions of photos we took. Some examples are included in this directory. It
works by optimising SGD on just one of the layers and propagating the
activations back to enhance the image. You can create various effects by
choosing the layer to augment, where higher layers tend to add higher-order
features to the image (mostly dog noses and eyes), and lower layers augment
local features (artistic effects).

