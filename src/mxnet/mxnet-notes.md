MXNet notes
===========

Note: COPY of internal notes from 2017-10-28

Notes while reading / working through The Straight Dope
in 2017-09+.
See also the mxnet-the-straight-dope repo.


# Basics
MXNet is written in C++ and runs efficiently on CPUs, GPUs, and distributed.
The performance scales almost linearly.
MXNet has a low-level interface, and a high-level interface, called Gluon.

It's on the edge of symbolic and imperative, with a dual API (NDArray and sym),
being able to both debug easily (imperative, interactive, but still running
efficiently on the GPU) with NDArray, or to create symbolic graph with sym that
can be compiled to even more optimised low-level code (eventually). Compilation
of the full computational graph at once can produce highly optimised code,
since it can reuse memory and simplify equations, and it will be cached and
never run twice with the same input. Network blocks could write generic code
accepting both types (with namespace input parameter)

All NDArray operations have a forward and backward function, making it possible
to easily derive the derivative automatically. MNNet has a smart scheduler that
will execute NDArray tasks non-blocking / async where possible; e.g., all
NDArray operators will return immediately and execute later inside MXNet (C++)
until the result is needed, then it is blocking (print, asnumpy(), etc).
The scheduler will also parallelise where possible, automatically modelling
dependencies, and execute pre-requisites in parallel where possible.
Can use arbitrary python control flow, which will be run inside python, and the
optimised graph will be a tree of possible branches in computations.

MXNet can run on multiple CPUs, multiple GPUs, or distributed over the network.
The Gluon Trainer does the necessary data synchronisation (split batch data,
add the gradients) between batches, but you can also write your own version.

To summary, scaling tricks in MXNet:
- run on multiple (x10) GPUs (x10), or multiple normal CPUs
- possible to run on a distributed cluster (x10?)
- Python interpreter, but math in highly optimised C++ library (10-100x)
- symbolic graph, thus can be simplified (x2)


# Architectures
Linear regression: 1 output node, with activation fn = identity function
Multi-class logistic regression: M output nodes on softmax activation
LeNet: 2x Conv + Pooling + FC + output nodes
AlexNet: 5x Conv with Pooling for (1,2,5) + 2x FC + output nodes
RNN: 1x FC with weights to output nodes + weights to itself


# Practical tricks
Some practical tricks:

- think about assumptions (data distribution, ..)
- experiment with hyperparameters (model complexity vs # data samples vs # dimensions)
- large improvements: batch normalisation, dropout (not needed when using batch norm.)


# Modules

**MXNet**, main functionality

    mxnet.cpu()/gpu()                           create context on CPU/GPU
    mxnet.metric.Accuracy()                     accuracy evaluator
        .update(preds=.., labels=..)


**NDArray**, by design very similar to numpy's ndarray, but with:

    1. automatic differentiation (operations have differentiated
       version / backward functions)
    2. async (non-blocking) CPU, GPU, and distributed cloud architectures;
       printing shows the location, e.g.: <NDArray 3x4 @cpu(0)>

    mxnet.nd
        .empty(<shape>)                         uninitialised memory
        .zeros(<shape>)                         ndarray initialised with zeros
        .random_normal(0, 1, shape=<shape>)     random gauss
        .array([[[values]]])                    specific (nested) arrays of values;
        .array([4.])                            scalar (needs to be in nd structure)
        .asscalar()                             convert (1D) to regular Python float
        .asnumpy()                              convert to numpy ndarray
        .arange([start,] stop)                  array range (non-lazy)
        .one_hot(<ndarr>, n)                    convert [0..n] scalar to one-hot encoding
        .save(<fname>, <ndarray>)               serialize to disk (more efficient than Pickle)
        .load(<fname>)                          deserialize from disk; full Block's / nets also
                                                have save_params(fn) and load_params(fn, ctx)

    NDArray
        .as_in_context(ctx)                     move to context (gpu/cpu) if needed
        .copyto(gpu(<n>))                       explicitly move data to be executed on gpu<n>
        .wait_to_read()                         don't execute async; wait for computation to finish
        .shape                                  get the shape (tuple of dims)
        .size                                   total number of values (prod of shape)
        .reshape(<shape>)                       re-arrange values
        .T                                      transpose
        +, -, *, /, ..                          normal elem-wise tensor operators
    nd.dot(x, y.T)                              regular matrix/tensor multiplication
    nd.sum(x[, axis=1])                         sum all elems (to scalar along axis)
    nd.argmax(<ndarray>, axis=N)                get the argmax
    nd.abs(x)                                   elem-wise abs(x_i)
    nd.norm(x)                                  euclidean length
    x = nd.exp(y)                               more ops; all return the new value
    x[:] = x + y                                re-use memory, but tmp variable
    nd.elemwise_add(x, y, out=x)                in-memory operations, no tmp variable



**autograd**, automatic differentiation;
important for neural nets because the loss function is differentiated so we
can update the weights in the right direction.
Similarly to PyTorch, the graph is built and differentiated on the fly based
on imperative code, and doesn't have to be compiled.

    x = nd.array([[1,2], [3,4]])                use ordinary ndarrays
    x.attach_grad()                             use autograd magic to record
    with autograd.record():                     start building graph, in train_mode;
                                                optional kwarg: train_mode=False
        y = net(x)                              e.g., forward pass in network
        z = loss(y)                             e.g., final cost function
    z.backward()                                backpropagate, filling .grad's


**sym**, symbolic graphs; mirror of NDArray, but symbolic computational graph

    mxnet.sym
        .var('data')                            return symbolic variable;
                                                can be feed into network / NDArray
            .tojson()                           print json representation of graph

    x = sym.var('data')
    y = net(x)
    y.tojson()
    y.list_arguments()                          variables that need to be provided
    y.list_outputs()                            output vars from this symbol node
    ex = y.bind(cpu(), {'x': nd.array(..)})     create executor
    ex = sys.Group([x, y]).bind(..)             create executor exposing multiple outputs
    ex.forward()                                execute symbolic graph

    Most of what's in NDArray, is also mirrored in sym (curr 285 out of 322),
    so both APIs can be used with the generic F namespace parameter,
    like inside HybridBlock.hybrid_forward(self, F, x)


**gluon**, high-level interface

    gluon.data.DataLoader(                      returns a batch iterator (NDArrayIter)
        gluon.data.ArrayDataset(X, y),
        batch_size=10, shuffle=True)
    gluon.data.vision.CIFAR10/MNIST             example datasets (live download)

    gluon.nn
        .Sequential()                           returns a sequential network (default Block)
            .add(<layer>)                       add a nn layer
            .collect_params().initialize(<init>, ctx=ctx)  like mx.init.Normal();
                                                for list of ctx'es, will be the same across devices
        .HybridSequential()                     simbolisable Sequential()
            .hybridize()                        compile to symbolic code (only compiles blocks
                                                that inherit from HybridBlock)
        .Dense(<outputs>)                       densely connected layer; etc
                                                only needs output size, infers input on first run;
        .Activation(activation='..')            activation layer, if not linear; most Blocks also
                                                accept kwarg activation='' ('followed by Activation')
        .Conv2D(channels, kernel_size)          CNN
        .MaxPool2D(pool_size, strides)          CNN pooling
        .Dropout(<ratio>)                       Dropout layer (only used in training phase)
        .BatchNorm(axis=1, center, scale=True)  Batch normalisation (_before_ expl. Activation layer)

    gluon.rnn                                   RNN layers; special since forward returns & expects state
        .RNN(<hidden>, <layers>)                create RNN layer
        .LSTM / .GRU                            create LSTM / GRU layer

    gluon.Block                                 inherit to create custom Blocks ((sub)networks); only
        .forward()                              need to define .forward() pass, can derive automatically;
                                                allows for run-time network architecture changes
        .save_params(f) / load_params(f, ctx)   (de)serialize to/from disk
    gluon.HybridBlock                           block that can be compiled; all standard gluon blocks
                                                inherit from this one
        .hybrid_forward(self, F, x)             need to define hybrid fw pass (F = mxnet.nd or .sym)
        .hybridize()                            compile to symbolic graph inside the c++ backend;
                                                can now input both sybolic and real ndarray data (which
                                                will be cast to symbolic automatically)
    
    gluon.loss.L2Loss()                         various loss functions
    gluon.loss.SoftmaxCrossEntropyLoss()        (incl Softmax activation layer)
    gluon.Trainer(                              returns a trainer (like SGD);
                                                automatically syncs params over multiple devices
        net.collect_params(), 'sgd', params)    p: learning_rate, wd, momentum, ..


**Parallelism**, general notes

    Multiple GPUs: use ctx = [gpu(i) for i in range(N)] everywhere; Gluon supports out of the box
    Multiple hosts: use mxnet.kv.create('dist') key-value store (sync NDArrays across network);
                    there is a tools/launch.py -H hostfile -n 2 python main.py  exec to help.

    nvidia-smi                                  CUDA command line tool: list GPUs and utility

    AWS has P2 instances with 1, 8 or 16 NVIDIA K80 GPUs ($6k/each; 40k parallel cores)
    + 192GB GPU memory + 4/32/64 vCPUs + 732GB RAM ($7/hour), in 2017

    Example runs of MXNet scaling linearly:
    === Run {dot 100x 4k x 4k} on GPU 0 til 15 in sequential ===
    time: 95.554450 sec
    === Run {dot 100x 4k x 4k} on GPU 0 til 15 in parallel ===
    time: 5.712788 sec

