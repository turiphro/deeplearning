# Week 10: Sunday, January 22, 2017

Today, we discussed the Deep Learning Book by Goodfellow, Bengio and Courville.
We got some cool insights that weren't mentioned in the Nielson book, and got
a closer look at recurrent neural networks (RNNs). After that, it was time for
cocktails, designed by IBM's Chef Watson.


## Deep Learning book
I read the majority of the Deep Learning Book, which consists of a general part
(I), practical deep learning networks (II), and more ideas currently explored
in the research literature (III). The book is recent (2016), very clear and
extensive, and has many more details than Nielson's introduction book.
Full notes can be found [in this file](../deeplearningbook_notes.md).

Part I is mainly a refreshment of math and the notation as used in the book,
and also features some practical issues with math on the computer. Underflow
(close to 0) and overflow ('close' to infinity) cause problems in practical
implementations, and therefore many implementations include quite some fixes
and small algorithmic changes (like using log() instead). Part I also has an
introduction in Machine Learning in general. Broadly speaking, one can design
new ML algorithms by making (mostly independent) choices for:
1. dataset specs
2. cost function (estimator + regularisation terms)
3. optimisation procedure (directly or iteratively like SGD)
4. model

Part II describes the basics on deep neural networks (1. 0-1 values,
supervised; 2. cross-entropy + regularisation; 3. minibatch SGD; 4. problem
dependent architecture, using RLUs). We've seen most of this already. It also
describes the algorithms of the large deep learning libraries, calculating the
computational graphs based on network descriptions. Some chapters follow with
more specific subjects:

* **Regularisation**: various tricks can be used to force certain types of
  solutions; we've seen all of them before. Dropout, weight penalty and
  dataset augmentation seem to be important.

* **Optimisation**: Machine Learning is optimisation on indirect targets
  (unseen test data). It requires generalisation tricks on top of normal
  (iterative) optimisation. Tricks on top of SGD are: adaptive learning rate
  (per dimension), second-order gradient approximations, momentum-based
  (similar to 2nd order approx), supervised or unsupervised pre-training
  (start on simpler problem or with simpler model, then scale up),
  continuation methods (iteratively solve less blurred version of cost
  function; similar to simulated annealing), and just using simpler models
  (RLUs are more linear, LSTMs are more stable, skip layers to prevent
  unstable gradient, etc).

* **CNNs**: we've seen these too; handy for data that has some local structure
  to exploit. They enable huge scaling improvements (orders of magnitude)
  since they share parameters and the number of connections scale much better.
  Pooling can happen over a small window (small translation-invariance), or
  over feature maps (allows learning other transformation invariance, like
  rotation or scale). The features learned on real-world data in the first
  layer(s) quite nicely resemble filters found in the human visual centre; they
  can sometimes also be reused in new models without retraining. CNNs can also
  handle varying sized inputs by adaptively changing the pooling window sizes.

* **RNNs**: We looked into the - slighly less well defined - group of models
  that share information over time. Biologically, this makes a lot of sense,
  and potentially opens up a world of new applications ('program approximation'
  instead of 'function approximation'). RNNs can be drawn as neural nets with
  loops, or 'unfolded' with a growing loop-less network over time
  (the computational graph). In this second representation, we can simply use
  backprop for training back 'over history', without algorithmic changes.
  
  There are a lot of possible architectures, like simply self-looping neurons
  over a sequence, or bidirectional (incorporates full context, incl. future),
  or encoder-decoder input RNN + output RNN pairs with a context ('meaning
  representation') in between, etc. Long-term dependecies are a big problem
  because of the unstable gradient problem in these hugely deep networks.
  Possible solutions include hierachical networks / computational graphs,
  LSTMs, and explicit memory (for knowledge-based systems with explicit facts).

* **Attention-based**: according to other resources, attention-based networks
  are currently hot in DL research; they usually combine data parsing networks
  (RNN or CNN) with a second network that 'looks at the data it needs'.
  Examples include looking at parts of images for QA systems, or translating
  while looking at words in the other language in different order.

* **Practical methodology**: some tips on practical matters; in reality,
  experience is needed to start developing real networks, and it's more
  important than picking the exact right network. Steps:
  1. determine your goals: error metric + target value
  2. get end-to-end pipelinen working ASAP (FNN/CNN/RNN + ReLU + SGD with
     momentum, decaying learning rate and early stopping; add regularisation)
  3. diagnose bottlenecks: overfit, underfit, bugs, data problem?
  4. repeatedly improve: more data, adjust hyperparameters (increase capacity,
     increase regularisation), change algorithms

* **Applications**: Mentions some applications with some background on pre-DL
  attempts and post-DL awesomeness. CV is super popular. Speech was late to the
  party. NLP is hard because of one-hot input and lack of distance function
  between words (little generalisation). Recommender systems have very sparse
  data too, but can tap into reinforcement learning algorithms, interacting
  with real users over time (cool!).

I did not yet read part III.


## Watson Cocktails
This was a lot to take in. Time to let Chef Watson design us some
**Cocktails**! IBM trained Watson with many recepes to learn - on a molacular
level, we're told - which ingredients mix well. There is a Chef Watson Twist
app that allows choosing some basic (cocktail) ingredients, with Watson
suggesting great additions. So there we went.

We got a clear winner with coconut juice, ginger beer, pineapple and tomato
ketchup (non-alcoholic category). The most polar opinions came from the
Campari + Beer + Egg white + Nutmeg, which somehow kept all the bitterness
but drove out the sweetness of the Campari.


## So long, and thanks for all the neurons
We hereby conclude our deep yourney into Deep Learning - for now. Thanks for
following along, and be careful while training your neurons.
