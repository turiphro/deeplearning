Deep Learning Book (2016)
-------------------------

    intro (1, 20p)
    part 1 ( 2-5,  160p): math & ML
    part 2 ( 6-12, 300p): DL practical
    part 3 (13-20, 230p): DL research


*Draft nodes written down while reading the Deep Learning Book by Goodfellow et al, 2016.*
*Note that these are just personal notes written as personal reference. They lack details*
*that have already been covered by the Nielson book (which I read previously), lack proper*
*typesetting and LaTeX evaluation, and may be brief and in telegram style.*

*I'm publishing them here anyway in case anyone finds them useful.*

General observations:

- very good book, clear and extensive, contains a lot
- advanced: start with simplier introductions (NN&DL book) first
- math-heavy, some sections might be skipped to see the broader picture
- read Nielson book (NN&DL) first, skipping over details that were explained there already



# ch1

* Deep learning is enabled by:
  - More data (1M-1BN): larger datasets AND digitisation
  - more compute power
  - various improvements in algorithms (learning + network architectures)

* Neural networks used to perform similarly to other (human-coded) AI systems, but start to
  vastly outperform other techniques for high-dimensional problems with huge datasets.

* DNN started with UL layered learning (RBMs), but now everyone is doing regular backprop
  with huge datasets (trading in model complexity for data).

* DNN are learning feature maps instead of need of engineering; can be unified model for many different AI subfields

* Neural nets are loosely inspired by the brain (mostly the architecture, not learning, since
  live scanning activity is still too hard) but modelling the brain is not the rigid goal
  (e.g., rectified linear not plausible but works the best, so far). The models are much
  simpler than human brains, but computers may compensate with more MIPS and more data.

* architectures:
  - fully connectede feed-forward (classic)
  - deep belief networks (layered unsupervised + total backprop; Hinton 2006)
  - CNNs (1998+)
  - RNNs

* convnets architecture: 80s, '98; based on visual cortex;
  feature maps = conv windows with shared weights for all pixels, constraing learned parameters by exploiting locality, allowing much higher resolution images



# ch2: LA

* scalars, vectors, matrices, tensors (ndarray)
* Ax = b  ->  x = A^-1 b, assuming A^-1 can be found (b \elem span(A))
* for orthonormal matrices: A^t A = A A^t = I and thus A^-1 = A^t
* norm_p(x) = (\sum |x_i|^p) ^ (1/p)  -> p=2: L2, euclidean distance; p=1: L1, simpler math; p=\inf: max(x)
* eigen decomposition: A = V diag(\lambda) V^-1, with V concatenated eigenvectors and \lambda the eigenvalues
* SVD: A = U D V^t, with U eigenvectors of AA^t and V eigenvectors of A^t A



# ch3: probability

* needed because of: inherent stochasticity in real world, incomplete observability, incomplete modeling
* discrete: \sum P(x=x) = 1, 0 <= P(x=x) <= 1
  continuous: \integral p(x)dx = 1, but p(x) >= 0 can be > 1, props of range only
* joint: P(x, y) = P(x|y)P(x)   = P(x)P(y) if indep -> x _|_ y
* marginalise: P(x) = \sum_y P(x, y),  p(x) = \integral p(x,y)dy
* chain rule: P(x, y1, ..., yn) = P(x) \prod_{i=2}^n P(yi | x, y1, ... y_{i-1})
* expectation (E) = avg;
  covariance(x, y) = E[ (f(x) - E[f(x)]) (g(y) - E[g(y)]) ]
* zero covariance (no linear dependence) != independence (no dependence)
* multivariate Gauss: N(x; \mu, \Sigma) = \sqrt(1 / ((2\pi)^n det(\Sigma))) exp( -1/2 (\vec{x} - \vec{\mu} \Sigma^-1 (\vec{x} - \vec{\mu}) )
  -> central limit theorem: sum of indep vars is approx N()
  mixture: P(x) = \sum_i P(c=x) N(X|c=i)
* information I(x) = -ln P(x)
  entropy H(x) = -E[ ln P(x) ] (expected amount of info for drawing from P)
  cross-entropy H(P, Q) = -E_{x~P} [ ln Q(x) ]
  kullback-leibner divergence D_{KL}(P||Q) = E[ ln P(x) - ln Q(x) ] (difference between P and Q)



# ch4: numerical math

* fundamental CS problems: underflow (close to 0 rounded to 0), overflow (rounded to \inf), poor conditioning (sensitivity to rounding errors). Reason for some algorithmic
  changes like taking log() inside cost optimisation, etc
* Gradient \/ f(x): vector with first partial derivatives
  Jacobian J f(x): matrix with first partial derivatives if f's output is vector
  Hessian H f(x): matrix with s econd partial derivatives (H is J of \/)
* gradient descent: move opposite direction of gradient.
  step size: fixed, or weighted using f'(x) or f''(x) or try a few per location
  Gradient descent is often used in optimisation problems if closed solutions are not possible,
  or for very large datasets (SGD).



# ch5: ML basics

* ML tasks: classification, regression, transcription, translation, structured
  output (parsing, segmentation), synthesis & sampling, denoising, probability
  distribution of data, ..
* designing a ML algorithm, choose (mostly independent):
  - dataset specification
  - cost function: typically statistical estimation (ML) + optional terms (regularisation)
  - optimisation procedure: solve directly, or approximate iteratively (SGD, etc)
  - model
* optimisation: training error is minimised
  generalisation: test data error is also minimised
  -> underfitting vs. overfitting trade-off  [3 graphs lin, _x^2_, x^9]
  -> depends on capacity of learning model (parameters etc) and amount of training data
     [training + generalisation error graph 5.3; add bias+var 5.6]
* regularisation: modify algorithm to reduce generalisation error but not training error;
  e.g.: add some penalty to cost function to optimise for certain preferences in
  solution space C(w) = MSE(w, X, Y) + \lambda w . w
* hyperparameters: not learned by algorithm (but can be by nested algorithms or
  validation data); e.g., controls model capacity, learning rate, etc
* one can use _estimators_ to estimate properties of learning algorithms based on
  the number of samples (bias and variance of the mean, or other model parameters);
  -> MSE estimator incorporates both bias and variance, thus optimising for both.
  -> maximum likelihood estimator: (frequentist statistics) take parameter that best
     explains all the data, to use for new predictions
  \theta_{ML} = argmax_\theta prod[ P_{model}(x^i; \theta) ]
              = argmax_\theta sum[log P_{model}(X^i;\theta)]
  -> maximum a posteriori: (bayesian statistics) use full distribution over possible
     parameters weighted by explainatory probability -> generalises better when little
     data, but higher computational cost for large training set; also includes prior
     (= regularisation term)
  \theta_{MAP} = argmax_\theta P(x|\theta) P(\theta)
               = argmax_\theta log P(x|\theta) + log P(\theta)
  In practice hard to calculate, thus approx with gradient descent.
* SL:
  - linear regression: P(y | x; \theta) = N(y; \theta^T x, I)
  - kernel trick: preprocess inputs with nonlinear function k (in fact a feature
    extractor); sometimes also more efficient to calculate than full dot product
    e.g., SVMs: f(x) = b + w . k(x^i; X)  -> uses all training data but only compares
    input data with the support vectors
  - k-nearest neighbor: non-probabilistic, non-parametric
* UL: learn data distribution, simplify data representation, clustering, etc
  -> representation learning algorithms: learn simpler representation:
     lower dimensional, sparse representation, independent representation 
  - PCA: rotates into space that has _linear_ correlations removed;
    X = U \Sigma W^T, take x^{i'} = \Sigma W^T x^i, possibly make \Sigma smaller
    for dim. reduction
  - k-Means: find k exemplar \mu means via iteration (assign data to \mu's, update \mu's)
* stochastic gradient descent (SGD): gradient descent with estimated gradient based on
  small batches (1-100s samples), _making training non-linear models with large datasets
  possible_, while most ML algorithms involve whole datasets in their multiplications

* Challenges motivating DL:
  - curse of dimonsionality: interesting problems have huge dim's, but generalisation is
    hard and needs exponentially more data
  - local constancy & smoothness (function) regularisation: by network representation
    -> but this alone is not enough for generalisation with small training set;
       DL adds function hierarchy (composition of features), thus combinatorial learning
       countering curse of dimensionality
  - manifold learning: assume most of R^n consists of invalid inputs, and learn only
    lower dimensional manifolds (non-linear subspaces) inside full space (1D road in 3D
    space; faces with changing light or rotation; ..) -> concentrate manifolds around
    the structure of the data (makes sense for many real-world tasks); seems to connect
    well with idea of hierarchy of concepts combined with holistic combinatorial learning.
    Every DL layer is basically a kernel trick, morphing the space to hopefully make it
    more linearly seperable. As with the kernel trick, we might need more neurons to
    increase dimensionality of the next layer's input. It is learning lower dimensional
    manifolds on top of the previous representation.



# ch6: deep feedforward networks

* Deep learning works great for vector->vector mappings that humans could do unconsciously.
  They aren't yet for tasks that don't have a simple mapping, or that require logic
  thinking or reflection.

* (Deep) neural networks are function approximators: y = f(x; \theta).
  They are loosely based on neuroscience, but now guided by engineering.
  Layers are like the kernel trick where we learn the kernel mapping \phi(x), which was
  constructed by humans before.
  - optimisation problem, solved using (stochastic) gradient descent
  - cost functions: MSE, cross-entropy (principle of maximum likelihood); + regularisation
  - activation functions: output layer is mostly dictated by cost function (cross-entropy:
    softmax), hidden layers are part of research (rectified linear is good default in FF,
    sigmoid in RNNs)
  - architecture: human-guestimated, then optimised with experimentation;
    universal approximation theorem states that any network with nonlinear activation fns
    can _represent_ any function (with certain approximation error), but adding layers saves
    exponentially more neurons, and it doesn't guarantee ability to _learn_ the function;
    architecture can be very different (convolutional, recurrent, etc).
  - learning: back-propagation (computes gradient using cost + activation fns) with SGD
    (updates parameters)

* Networks can be reasoned about as computational graph, with nodes for variables (tensors)
  and edges for operators. Graphs can then (symbolically) be simplified, evaluated or
  differentiated in new graphs (e.g., TensorFlow, Theano). Alternatively, direct numerical
  calculation can be done (e.g., Torch, Caffe). -> explanation of algorithms inside DL libs


# ch7: regularisation

* Various tricks to improve generalisation: encode prior knowledge, preferences for simpler
  models, making underdetermined problems determined, or combining hypotheses.
  In deep learning finding simpler models is often mandatory, since we're fitting on very
  high dimensions for which we can't possibly find the real data distribution (model).

* Tricks for DL:
  - parameter norm penalties: add parameter size penalty to cost function; usually on
    weights only; has effect of slowly decaying weights on every update
  - making underconstrained problems constrained
  - dataset augmentation: simple transformations on or adding noise to data; highly
    problem-dependent but can cause huge improvements
  - early stopping: use parameters of lowest validation error, stop when not improving
    (kind of treating # epochs as a hyperparameter to learn)
  - parameter sharing: force sets of parameters to be equal (convnets); allows dramatic
    increase of network sizes for same amount of data (less parameters to learn)
  - sparse representations: most neurons shouldn't fire; added to cost function;
    biologically inspired
  - ensemble: averaging independently trained (possibly quite different = boosting) neural
    networks, possibly trained on different data subsets (bagging); very powerful and
    reliable
  - dropout: disable half of input and hidden neurons for each mini-batch; makes
    representation more robust and holistic; kind of bagging for many networks at once but
    with shared parameters; kind of adding noise to hidden layers; very cheap and generic
    (FFD, RNNs, Boltzmann) on medium-sized datasets, but does need a bigger model and more
    training


# ch8: optimisation

* Machine learning is optimisation on indirect targets (`P`: minimising error on unseen
  test set). We optimise some cost function and hope it actually optimises `P`. The
  generalisation part for unseen data (don't overfit with high capacity of model) makes
  it ML instead of direct optimisation. We therefore optimise some loss function that is
  a surrogate for what we actually care about. Usually not solvable in closed form, thus
  using iterative solvers like SGD, and halting when a (possibly different) loss function
  on the _test_ data stops improving (e.g., the actual classification accuracy).

* Challenges in neural network optimisation: ill-conditioned, local minima (non-convex
  cost function), other flat regions (actually exponentially more common in high-D, more
  dims -> more chance one will go down -> in practice local minima have low cost; use
  second deriv), cliffs & exploding gradients (due to multiplications; use gradient
  clipping; in RNNs use LSTMs), inexact gradients (because minibatch estimation), no
  minimum at all (certain cost functions), very long/inefficient local SGD paths and
  different scales of structures.

* Algorithms:
  - SGD: gradient descent with minibatches (gradient estimate for each step), with
    decreasing learning rate
  - momentum-based SGD: accelerate learning at places of high curvature or consistent
    or noisy gradients; can be seen as numerical physical simulation of movement that
    estimates the second derivative of the cost function
  - adaptive learning rate (often per dimension = parameter) SGD: AdaGrad, RMSProb, Adam
  - second-order gradient approximators: Newton's Method (very expensive O(k^3), Hessian),
    conjugate gradients, BFGS (iteratively estimate Hessian^-1)
  - other tricks: batch normalisation (solves not-so-independent-updates problem);
    coordinate descent; polyak averaging (average visited locations to find valley);
    supervised pretraining (start on simpler problem or with simpler model; e.g., subset
    of all layers, or add layers in between); continuation methods (iteratively solve
    less blurred versions of cost function; similar tosimulated annealing); just choose
    simpler models (RLUs, LSTMs, connections that skip layers).

* Parameter initialisation in NNs is poorly understood and mainly heuristic. It must break
  symmetry (nodes must be different) though: random weights, not too small (optimisation
  says: promote node specialisation) and not too big (regularisation says: unstable or
  node saturation). Sometimes ML is used to initialise (e.g., UL with Boltzmann machines).


# ch9: CNNs

* Specialised kind of NN for grid-like data (time-series, images, voxels, video,
  kinematic angles), that uses a convolution-like function instead of fully connected in
  at least one layer. "Shared weights over space." CNNs are highly motivated by
  neurological observations. They have been incredibly powerful for 20 years on large
  problems, but went unnoticed for a long time. Note that the usefulness is highly
  dependent on the problem, which should have a local (and hierarchical) structure.
  
  CNNs leverage three important ideas:
  1. sparse interactions: exploit localised information; enables huge scaling (orders of
     magnitude): O(m.n) -> O(k.n) runtime; information can still reach all output neurons,
     but travels (much) slower outwards towards deeper layers.
  2. parameter sharing: reuse kernel (window function) for all neurons; weights and usually
     biases too: O(m.n) -> O(k.n) parameters = storage and statistical efficiency
  3. equivariant representations: translation invariant (but not scale or rotation)

* **Convolution**: combine two functions: move a window function over the actual function
  while integrating (or weighted averaging when discrete);
  s(t) = (f * w)(t) = \int f(a).w(t-a)da;
  Can be used for averaging, smoothing, finding templates, etc.

* Practical networks often alternate between:
  - convolution (linear): usually multiple convolutions (feature maps) in order to learn
    multiple features on this level in the hierarchy, with a tensor (2D or 3D) as input
  - activation function (nonlinear; 'detector')
  - pooling (summarise/downsample; 'infinitely strong prior' on weights (zero outside of
    window, shared window weights along input nodes); can also be used to process images
    of variable size): function to combine neighbouring nodes, like `max` or `avg`;
    1. if run over window: gives small-translation invariance 
    2. if run over multiple feature maps: gives other learned transformation invariance,
       like rotation or scale

* Various notes:
    - stride: skip input pixels (kind of downsampling)
    - zero padding (valid, same or full): handle border
    - unshared convolution / locally connected layers: no shared parameters but
      does have a convolution as structure
    - generative or inverse actions need some tricks for (strided) convolutions and
      for pooling

* CNNs can process varying size inputs, since the kernel simply scales to larger inputs.
  It can either have varying size outputs too, or pooling layer regions that scale with the
  input size.

* Structured (tensor) outputs can be made with convolutional layers (pixel classification,
  region segmentation), possibly using recurrent elements to use the network as graphical
  model.

* Speed-ups:
  - evaluate using fourier (input, kernel) + multiply + inverse fourier (in some cases)
  - learn feature maps first, independent from rest of (very deep) network: set random
    weights, hand-craft features, unsupervised learning (e.g., k-means on patches, then
    SL with pooling and other upper layers only), or layer-wise SL pretraining with some
    hidden layer goal (e.g., conv deep belief network)
  In practice it's mostly full SL training now though, and UL pre-training is
  under-explored.

* CNNs have their basis in old neuroscientific research (1960s+), mimicking some properties
  of the visual system of the human brain. They also act similarly to time limited humans
  doing object recognition (but not later stages, where concepts flow backwards again and
  more cognitive tasks are performed). There are big differences and unknowns though
  (steered quick attention areas, combined senses, 3D geometry, feedback loops, actual cell
  behaviour, and actual training algorithms). In particular, backpropagation seems
  biologically implausible. Simple cells in lower human conv layers (V1) seem to activate
  on Garbor-like functions, while complex cells seem to compute the L^2 norm. Almost all
  ML techniques on natural images also learn Garbor-like features (edge detectors).
  

# ch10: RNNs

* RNNs are various network structures for processing sequential data (language, sound,
  annotating data, QA, sequentially reading fixed-size data), and can usually process or
  produce sequences of arbitrarily length.
  They contain information loops, allowing the network to have a memory over time:
  "Sharing weights over time." With added memory, they're more like "program
  approximators" than "function approximators". It stems from ideas from the 1980s.
  - each hidden layer h, given sequence x: h^t = f(h^{t-1}, x^t; \theta)
  - various architectures are possible: connections within hidden layer(s); connections
    from output to hidden layer(s) = less powerful but trainable with ground truth (y);
    connections within layer producing one summary after full sequence; output back to
    input data = autoencoder
  - various visualisations: neural network (nodes output to themselves), or unfolded
    computational graph (nodes output to next node in time sequence of len(x) for each 
    node)
  - parameter sharing (over time) means we need much less data (like with CNNs)

* Training:
  - loss: sum of losses over full sequence
  - gradient (BPTT: 'backprop through time') is expensive & not parallelisable: backprop
    through full sequence of neurons (regular backprop on unrolled computational graph).
    Except for output to hidden layer recurrance (teacher forcing: train with expected
    network output instead of actual output).

* much computational graphs, many distributions. Rather abstract without any applications.

* Problems with RNNs:
  - unstable gradient problem, for long sequences; more data!
  - long-term dependencies: remember state needed much later in a sequence; partly caused
    by unstable gradient (many multiplications), and by lack of long-term information
    storage (everything vanishes slowly). Remains one of the main challenges in DL.

* Some more architectures that are being used:
  - **bidirectional RNNs**: outputs depend on the full sequence (previous, current, *and*
    future inputs); combine an RNN moving forward and one moving backwards through time;
    speech recognition, handwriting, bioinformatics, etc. (Unclear how they are actually
    run.) Also possible for 2D inputs (images) with 4 RNNs (up, down, left, right)
  - **encoder-decoder RNNs**: variable-length input to variable-length output through
    encoder to context/state through decoder; useful in speech recognition, machine
    translation, QA systems, etc
  - **deep recurrent networks**: recurring connections can be used in deeper networks as
    well, on various places.
  - **recursive RNNs**: the computational graph is a tree instead of a chain, thus
    most variables have a much shorter (log n) dependency path, helping for long-term
    dependencies; useful for data structures, NLP, and CV
  - **Multiple time scales**: skip connections across time (also connect h_t with h_{t+k}),
    **Leaky Units** (linear self-connection = running average), removal of connections
    (only connect h_t with h_{t+k})
  - **LSTM, gated RNNs (GRUs)**: the main idea is to add specific subnetworks with specific
    behaviour that can be used by the rest of the network. LSTMs are elements consisting of
    a state with self-loop, and specific neurons for setting, forgetting (time-scale
    parameter), or getting the value; network learns to use the gates (unclear how); used
    in handwriting generation and recognition, machine translation, parsing, image
    captioning. So far the **most effective solution** in RNNs, and used almost explicitly.
  - **Explicit memory**: some knowledge is better saved explicitly (constants, hierarchies,
    names); adds memory cells with (soft) addressing mechanism to the network. Taking a
    weighted average of all cells makes it differentiable (SGD), and allows for both fuzzy
    content-based (dot-product all items) and index-based (sequential) search. Used in
    neural Turing machines (NTMs). Current research tries to move to hard addressing: much
    cheaper, but not differentiable; using RL techniques seems promising.
  - **Attention-based approaches**: current research looks at letting the network decide
    which data to look at (in large corpus, during audio annotation, which part of an
    image). Works well for combining multiple networks (e.g., CNN processing an image, then
    an RNN annotating when looking at parts of CNN's output). It seems biologically
    inspired again, with potentially huge gains in processing speed and accuracy.

* Sampling from an RNN allows for "dreaming" up likely sequences: start random, and keep
  sampling from the softmax output (likely next items in sequence), feeding it back into
  the network. Given enough data, it can actually learn language on character-level,
  markup structures, and even syntactically correct (but non-compiling) code.


# ch11: practical

* The most important ML skills to learn are practical matters. Informed guessing what to do
  to improve results is often more important than the specific algorithm or architecture.
  It is an art improved over time by gaining experience. A good approach consists of:
  
  1. **Determine your goals**:
     Error metric (accuracy, precision+recall or F-score, coverage (% not classified with
     certainty), custom metrics like click-through rates) + Target value
  2. **Get end-to-end pipeline working ASAP**:
     Start with baseline models without DL (if not "AI-complete") and vanilla
     implementation of FFN (fixed-size input vector), CNN (topological structure), or
     gated RNN (input or output sequence). Use ReLUs, SGD with momentum and decaying
     learning rate, and early stopping. Try batch normalisation. Add regularisation (if
     less than 10M samples). Try models trained on similar problems, try even the trained
     model (e.g., feature maps from ImageNet CNN). Try unsupervised learning if relevant
     (NLP).
  3. **Diagnose bottlenecks**: overfit, underfit, bugs, data problem?
  4. **Repeatedly make incremental changes**
     - *more data*: often more important than trying more algorithms; but only once results
       on the training data are OK but poor on the test set (generalisation). Plot training
       set size (log) vs generalisation error to estimate amount of data needed.
     - *adjust hyperparameters*: trade-off between manual (thorough understanding) vs
       automatic grid-search (training time).
       1. Manual: match effective model capacity to complexity of the task;
          most hyperparameters cause a U-shaped generalisation error (underfit <> overfit).
          Start with the learning rate. Increase model capacity (neurons, layers, conv
          kernel width) until happy with training error. Then increase generalisation by
          adding/improving regularisation (dropout, weight decay) and/or more data.
       2. Automatically: when no starting point is available, do rough search;
          grid search for small set of parameters (search all combinations), usually on
          log scale, repeated on multiple scales; random search for more parameters (sample
          parameter values from distribution, possibly changing the distribution based on
          the results); and experiments are being done with automatic hyperparameter
          optimisation algorithms building hyperparameter models and performing experiments
          (bayesian; reinforcement learning?).
     - *change algorithms*

* **Debugging** machine learning systems is hard, because we don't know what's the "correct"
  behaviour. Difference between bugs and bad performance is hard to spot, also because
  the ML algorithm's goal is to adapt to any inefficiencies (or bugs).
  Some strategies: visualise the model & results in action (not only the error metric),
  visualise the worst mistakes (using confidence estimate; find preprocess or label
  problems); look at train/test error abnormalities; test with tiny datset; double-check
  your gradient implementation; visualise avg (batch) activation histograms per neuron /
  parameter (saturation, always off, odd magnitudes or gradients)


# ch12: applications

* Real-world applications need large-scale networks, executed code optimised for general
  purpose GPUs (high parallelism, high memory bandwidth) - using existing DL libraries -
  run on large-scale distributed clusters. GP-GPUs (which can execute arbitrary C code,
  like NVIDIA's CUDA platform) are relatively new (2007+). Recently also DL-specific
  hardware (NVIDIA, Google) or FPGA/ASIC implementations are being used. SGD can partly
  be parallelised, but there are limits to the scale.
  Final learned models might be compressed for low-end inference devices.

* **Computer vision**: most popular DL field; mostly object recognition or
  detection, and some image generation. Not a lot of preprocessing is needed (normalise to
  [0,1], and normalise contrast, globally or locally). Heavily uses CNNs.

* **Speech recognition**: until 2012 mostly HMMs (phoneme sequence) + GMMs (acoustic
  features <> phonemes), then switching to RBMs (2009), then regular DL networks (2013),
  both CNNs and RNNs, initially combined with HMMs.

* **NLP** (machine translation, transcription, QA): usually based on language probabiltiy
  distributions, using word-based (high-D in and out) RNNs; incorporate some
  domain-specific strategies. NLP classically creates n-grams, where the ML is based on
  counting, plus some solution to the problem that most will be 0.
  Neural language models: distributed word representation (word embeddings); the first
  hidden layers are much more important than in other domains (input is discrete, one-shot
  with no similarity measure). High-D output also needs some solution since it's so
  expensive (hierarchical softmax, importance sampling (during training, only update some
  words' neurons)).
  - Machine Translation: moved from n-grams to neural language models; in particular,
    encoder-decoder RNN network, yielding a target sentence conditioned on the source
    sentence via the (learned) context representation. Research focusses on
    attention-based systemes: reading the full sentence, saving feature vectors in
    memory, and then translating by looking more specifically using the memory.

* Other:
  - **Recommender Systems**: traditionally user-item pairs, embedding representations
    learned (SVD) and dot-producted to obtain similarity; later ensembles which include
    RBMs, and deep neural nets for learning content-based feature vectors (embeddings)
    to compare. Needs a cold-start strategy. On long-running systems, there's the
    Exploration-Exploitation trade-off that makes reinforcement learning a useful
    approach (exploration can be random recommendations, or based on uncertainty of
    expected reward).
  - **Knowledge representation, QA**: incorporating and using logic (binary facts) in
    neural networks, by learning embedding vectors for relations (entity,relation,entity).
    Evaluation is hard: we only have positive facts (can't evaluate the outputs). Possibly
    combined with (traditional) reasoning systems, and explicit memory. Still in its
    infancy.


# Part III

Not read (yet). More experimental research areas. Smarter algorithms that need less data
and/or work on multiple domains.


