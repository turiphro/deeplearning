This Sunday (Octobre 2) we started out discussing chapter 1 and 2 of
the Nielson book. The basics of neural nets involve:

- the **neuron**: summing weighted inputs and putting that through a certain
  function (step, sigmoid, rectified linear, softmax, etc). The neuron has
  learnable parameters for input weights and one (activation) bias:
  `a = f(w . x + b)`
- the **network**: layers of neurons, usually fully forward-connected,
  propagating activation values forward until the output layer
- the **learning**: backpropagation with stochastic gradient descent,
  taking batches of training data, calculating output errors, and propagating
  the error back to earlier layers to update the parameters (weight, bias)

Next session (chapter 3) will be about a bunch of possible improvements
to this basic scheme. The session thereafter (chapter 4 + 5) involves some
intuition on learning, and the final session on this book (chapter 6) gets
to the meat of deep learning. After that we will start with practical
projects and reading up on specific deep learning subjects.

We also watched a 20min talk by Hinton (https://www.youtube.com/watch?v=l2dVjADTEDU)
and discussed why deep learning works now but didn't in the 80s: more data
(and public datasets), more compute power (CPU, GPU), and tweaks in the
algorithms (both learning and structure). There are ways of structuring the
networks in order to learn better for specific data (convolutional: images;
recurring: time sequences; autoencoders, etc).

Martijn also showed his demo app (https://github.com/gmtahackers/deeplearning/tree/master/src/visual-tester)
for real-time classification of webcam streams using Nielson's vanilla
implementation of neural nets. It can work reliably when holding digits
drawn on paper, but only when aligned well (e.g., during the demo, the
network classified Martijn's face as '2' with high certainty).

So next session: chapter 3! And bring your pet project.
