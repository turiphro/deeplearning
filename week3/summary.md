2016-10-16, by Martijn

The third session dealt with chapter 3 of the NN&DL book. Quite some
interesting points came up during discussions of the various
improvement proposals, and some questions stayed unanswered as the
book glances over a lot of theory.

There is a large number of improvements to the vanilla implementation
of neural networks, including:

- different **cost functions**, which might increase learning rates
  for neurons that are quite wrong, e.g., cross-entropy (sigmoid +
  LSE slow down learning when the activation is close to 0 or 1)

- different **activation functions**: the final layer might use `softmax`
  so we can interpret the activations as probabilities; we discussed
  `softplus` as well (softened version of `max()`) but couldn't find
  reasons for preferring one or the other. One can also use the rectified
  linear activation function (`rulu`) for all neurons, it being almost
  linear while the network is still able to learn non-linear functions.
  However, in our discussion it turned out the `sigmoid` is likely the
  most accurate model of real neurons, while we're preferring more linear
  functions because of the unnatural backpropagation learning algorithm.

- use **regularisation**: preferring certain solutions above others in order
  to prevent overfitting (= better generalisation), often "simpler"
  solutions. This might happen by adding a **penalising term**
  to the cost function (`L1`, `L2`, or others) to keep weights low
  (prevents overfitting on the noise), or changes to the learning
  algorithm like **dropping**, whereby some neurons are temporarily
  deactivated for each batch (prefers more robust and 'holistic'
  network).

- change **weight initialisation**: better to initialise with smaller
  distribution, e.g., 1/sqrt(n_in)

- additions to **gradient descent**: calculate (Hessian) or estimate
  (momentum) the second derivative to change the learning steps;
  variable learning rates

- artificially **expand the training data**: as the networks are
  sensitive to transformations like rotations, intensity, and translation
  (except CNNs), we can generate more data by transforming the existing
  data. Carefully choose the transformations though. Basically, we're
  compensating the simplicity of the model with more data.

- **hyper-parameters guestimating**: some strategies in chosing the
  hype-parameters that make it easier to get results quicker include:
  simplifying the problem and network until we get at least _some_
  results, as it often happens we don't get anything (which makes tuning
  hard); play with the learning rate (e.g., oscillation); early stopping
  for epochs

After having fun with the laser-cutters, we played with **Scikit-learn**
(sk-learn), a wildly popular machine learning library in python. We tried
the MNIST dataset on the standard neural network, which is not deep but
does have som of the improvements from above (0.97% accuracy). Using this
broad framework also makes it easier to compare neural networks with other
algorithms, like SVM (also 0.97% accuracy).

Next session will include an introduction into **TensorFlow**, so we can
go deeper into the networks.
