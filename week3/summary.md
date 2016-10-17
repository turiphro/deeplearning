2016-10-16, by Martijn

The third session dealt with chapter 3 of the NN&DL book. Quite some
interesting points came up during discussions of the various
proposals for improvements, and some questions stayed unanswered as the
book glances over a lot of theory quickly.

There is a large number of improvements to the vanilla implementation
of neural networks, including:

- different **cost functions**, which might increase learning rates
  for neurons that turn out to be quite wrong, as the sigmoid +
  LSE slow down learning when the activation is close to 0 or 1;
  e.g., cross-entropy cost function

- different **activation functions**: the final neuron layer might use
  `softmax` so we can interpret the activations as probabilities; we discussed
  `softplus` as well (softened version of `max()`) but couldn't find
  reasons for preferring one or the other. One can also use the rectified
  linear activation function (`ReLU`) for all neurons, it being almost
  linear while the network is still able to learn non-linear functions.
  However, in our discussion it turned out the `sigmoid` is likely the
  most accurate model of real neurons, while we're preferring more linear
  functions just because of the unnatural backpropagation learning algorithm.

- use **regularisation**: preferring certain solutions above others in order
  to prevent overfitting (= generalise better); often we want "simpler"
  solutions. This might happen by adding a **penalising term**
  to the cost function (`L1`, `L2`, or others) to keep weights low
  and prevent overfitting to the noise; or make changes to the learning
  algorithm like **dropping**, whereby some neurons are temporarily
  deactivated for each batch, which turns out to prefer more robust and
  'holistic' network.

- change **weight initialisation**: it's better to initialise with smaller
  distribution, e.g., 1/sqrt(n_in)

- additions to **gradient descent**: calculate (Hessian) or estimate
  (momentum) the second derivative to change the size of the learning steps;
  use variable learning rates

- artificially **expand the training data**: as the networks are
  sensitive to transformations like rotations, intensity, and translation
  (except in CNNs), we can generate more data by transforming the existing
  data. Carefully choose the transformations though. Basically, we're
  compensating for the simplicity of the model with more data.

- **hyper-parameters guestimating**: some strategies in chosing the
  hype-parameters that make it easier to get results quicker include:
  simplifying the problem and network until we get at least _some_
  results, as it often happens we don't get anything (which makes tuning
  hard); play with the learning rate (e.g., oscillation); early stopping
  for epochs when cost doesn't improve (so we can retry quicker)

After having fun with the laser-cutters, we played with **Scikit-learn**
(sk-learn), a wildly popular machine learning library in Python. We tried
the MNIST dataset on the standard neural network, which is not deep but
does have som of the improvements from above (0.97% accuracy). Using this
broad framework also makes it easier to compare neural networks with other
algorithms, like SVM (also 0.97% accuracy).

The next session will include an introduction into **TensorFlow**, so we can
go deeper into the networks.
