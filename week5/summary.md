2016-11-13, by Martijn

For the final step towards practical deep learning, we'll need to talk about
structure. Instead of the fully connected deep forward networks, we can also
use different structures in neurons and connections. We discussed convolutional
neural networks, as described in the book, and watched videos describing other
alternatives as well.

* **Convolutional** (CNNs): exploiting locality (images, sound `x` frequency
  grid) by connecting neurons to a *small window* only of neurons in the former
  layer. We therefore keep the neurons arranged in a grid (instead of a 1D
  array) and limit the number of connections drastically. Furthermore, all
  weights between the windows for all pixels in one layer are shared. This
  setup will allow the network to learn its own *feature maps*, where AI
  researches previously had to come up with clever features to extract from
  raw data.

  From each `n x n` layer (plus padding) we can create `k` of these feature
  maps, resulting in a `k x n x n` tensor. The convolutional layers are usually
  followed by pooling layers, which down-sample the image - this is in fact
  another convolution with functions like `max`, while skipping some input
  rows and columns for the down-sampling. These two layers are repeated multiple
  times, and followed by one or more fully connected layers, summarising the
  results. This setup works really well for image-related problems, as proven
  on the MNIST, ImageNet, and other datasets.

* **Recurrent** (RNNs): breaking with the forward networks, these structures
  get input from previous activations, and work well on time-varying
  sequences like natural language understanding or production, both text and
  in sound format. They are trained with a variant of back-propagation,
  although they have an even bigger instable gradient problem.

* **Long Short-Term Memory** (LSTMs): they are a solution for the instable
  gradient problem in RNNs. They are not explained.

* **Boltzmann machines** (RBMs): the original breakthrough leading to *deep*
  learning was using a restricted Boltzmann machine (input + one hidden layer)
  in unsupervised training, activating the hidden layers to 're-generate' the
  original input, using the difference to improve the weights. This forces
  learning of good representational hidden neurons. Stacking RBMs on top of
  each other (called **Deep Belief Networks**) gives a new way of training
  deep neural networks. Once layer-by-layer training is done, the final result
  can be trained using the final desired outputs (supervised learning). We
  could therefore do semi-supervised learning, training on the vast amounts of
  unlabelled data out there and only labelling a small percentage. They can
  also be used in a generatively fashion, generating new data based on the
  learned structure in real data. Despite much promise, RBMs are currently out
  of fashion.

  RBMs are closely related to **Auto-encoders**, which usually consist of
  encoding data (input -> hidden) and decoding (hidden -> output) networks,
  while training on the difference (input / output). 

* **Reinforcement learning**: some research (e.g., Google DeepMind) focusses
  on combining DNNs with reinforcement learning, allowing to learn from
  experience by interacting with the environment. This seems still mostly
  a field of research.
