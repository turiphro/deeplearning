import mxnet as mx
from mxnet import nd, autograd, gluon
from matplotlib import pyplot as plt

# linear algebra
s = nd.array([8.])                      # scalar
x = nd.arange(4)                        # array
X = nd.ones((3, 3))                     # matrix
Y = nd.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Z = X * Y
print("Z[1:2,1:2] = ", Z[1:2, 1:2])
T = nd.arange(24).reshape((2, 3, 4))    # tensor
print("T . x =", nd.dot(T, x))          # most inner dim of T dot former most dim of x
nd.norm(x)                              # L2
nd.sum(nd.abs(x))                       # L1


x = nd.array([[1, 2], [3, 4]])              # use ordinary ndarrays
x.attach_grad()                             # use autograd magic to record
with autograd.record():                     # start building graph
    y = x * 2
    z = y * x                               # final cost function
z.backward()                                # backpropagate

print("x.grad:", x.grad)


## Linear regression: let's estimate this fn
ctx = mx.cpu()


def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

# network: 1 node, identity activation fn
n_inputs = 2
n_outputs = 1
n_examples = int(1e4)
lr = .001
X = nd.random_normal(shape=(n_examples, n_inputs)) # random inputs
noise = .01 * nd.random_normal(shape=(n_examples,))
y = real_fn(X) + noise

## FROM SCRATCH
w = nd.random_normal(shape=(n_inputs, n_outputs))  # random initialised weights
b = nd.random_normal(shape=n_outputs)
params = [w, b]
for param in params:
    param.attach_grad()


def net(X):
    return mx.nd.dot(X, w) + b


def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)


def sgd(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


## OR WITH GLUON:
# The network is initialised on the first forward pass; this is lazy setup
net2 = gluon.nn.Sequential()        # define network
with net2.name_scope():
    net2.add(gluon.nn.Dense(n_outputs))# optional (lazily inferred): in_units=n_inputs
net2.collect_params().initialize(   # random initialised weights (lazy eval)
    mx.init.Normal(sigma=1.), ctx=ctx)
square_loss2 = gluon.loss.L2Loss()  # returns function
sgd2 = gluon.Trainer(net2.collect_params(), 'sgd', # define optimiser; incl momentum
                     {'learning_rate': lr})


# let's train!
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                   batch_size=batch_size, shuffle=True)


def train(train_data, net, loss_fn, trainer, ctx, epochs=2):
    losses = []
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx) # ?
            label = label.as_in_context(ctx).reshape((-1, 1))
            with autograd.record():                     # execute graph
                output = net(data)                      # forward pass
                loss = loss_fn(output, label)           # calculate loss
            loss.backward()                             # get deriv's (set .grad's)
            trainer()                                   # update weights

            losses.append(nd.mean(loss).asscalar())
            if (i+1) % 500 == 0:
                print("Epoch {:3d}, batch {:5d}. Curr loss: {:2.10f}".format(
                    e, i, nd.mean(loss).asscalar()))
    return losses


def plot_result(losses, X, net, real_fn, sample=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample, 1].asnumpy(), net(X[:sample, :]).asnumpy(), 'or')
    fg2.plot(X[:sample, 1].asnumpy(), real_fn(X[:sample, :]).asnumpy(), '*g')
    plt.show()


# Manual way:
losses = train(train_data, net, square_loss, lambda: sgd(params, lr), ctx)
plot_result(losses, X, net, real_fn)

# Gluon way:
losses = train(train_data, net2, square_loss2, lambda: sgd2.step(batch_size), ctx)
plot_result(losses, X, net2, real_fn)

