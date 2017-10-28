import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np

# Let's classify MNIST digits using a multi-class logistic / softmax /
# multinomial regression: 10 nodes with softmax activation fn and
# cross-entropy loss fn


def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)


def transform3d(data, label):
    return (nd.transpose(data.astype(np.float32), (2, 0, 1))/255,
            label.astype(np.float32))


num_inputs = 28*28
num_outputs = 10
batch_size = 64

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test  = gluon.data.vision.MNIST(train=False, transform=transform)
train_data  = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data   = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

mnist_train3d = gluon.data.vision.MNIST(train=True, transform=transform3d)
mnist_test3d  = gluon.data.vision.MNIST(train=False, transform=transform3d)
train_data3d  = gluon.data.DataLoader(mnist_train3d, batch_size, shuffle=True)
test_data3d   = gluon.data.DataLoader(mnist_test3d, batch_size, shuffle=False)

ctx = mx.cpu()
learning_rate = .1


## MANUAL VERSION

W = nd.random_normal(shape=(num_inputs, num_outputs)) # one prototype vector per class
b = nd.random_normal(shape=num_outputs)
params = [W, b]
for param in params:
    param.attach_grad()


def softmax(a):
    exp = nd.exp(a-nd.max(a))
    norms = nd.sum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / norms


def net(X):
    a = nd.dot(X, W) + b
    yhat = softmax(a)
    return yhat


def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat), axis=0, exclude=True)


def sgd(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


## GLUON VERSION

# no softmax layer needed: gluon has loss fn with softmax built in
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# simple network
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(num_outputs))

net2.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)

sgd2 = gluon.Trainer(net2.collect_params(), 'sgd',
                     {'learning_rate': learning_rate})

# alternative network
net3 = gluon.nn.Sequential()
num_hidden = 256
with net3.name_scope():
    net3.add(gluon.nn.Dense(num_hidden, activation='relu'))
    # Dropout blocks are used only when in train mode (default in autograd.record() scope)
    # Dropout slightly decreases test data accuracy, but decreases the generalisation error
    net3.add(gluon.nn.Dropout(.5))
    net3.add(gluon.nn.Dense(num_hidden, activation='relu'))
    net3.add(gluon.nn.Dropout(.5))
    net3.add(gluon.nn.Dense(num_outputs))

net3.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

sgd3 = gluon.Trainer(net3.collect_params(), 'sgd',
                     {'learning_rate': learning_rate})

# CNN network
net4 = gluon.nn.Sequential()
num_fc = 512
with net4.name_scope():
    net4.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net4.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net4.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net4.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net4.add(gluon.nn.Flatten()) # back to 1D for fully connected layers
    net4.add(gluon.nn.Dense(num_fc, activation='relu'))
    net4.add(gluon.nn.Dense(num_outputs))

net4.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

sgd4 = gluon.Trainer(net4.collect_params(), 'sgd',
                     {'learning_rate': learning_rate})

# CNN network with improvements
net5 = gluon.nn.Sequential()
num_fc = 512
with net5.name_scope():
    net5.add(gluon.nn.Conv2D(channels=20, kernel_size=5))
    net5.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net5.add(gluon.nn.Activation(activation='relu')) # need explicit, AFTER BatchNorm
    net5.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

    net5.add(gluon.nn.Conv2D(channels=50, kernel_size=5))
    net5.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net5.add(gluon.nn.Activation(activation='relu')) # need explicit, AFTER BatchNorm
    net5.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

    net5.add(gluon.nn.Flatten()) # back to 1D for fully connected layers

    net5.add(gluon.nn.Dense(num_fc))
    net5.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net5.add(gluon.nn.Activation(activation='relu')) # need explicit, AFTER BatchNorm
    net5.add(gluon.nn.Dense(num_outputs))

net5.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

sgd5 = gluon.Trainer(net5.collect_params(), 'sgd',
                     {'learning_rate': learning_rate})


def evaluate_accuracy(data_iter, net, reshape=True):
    acc = mx.metric.Accuracy()
    #num = den = 0.
    for i, (data, label) in enumerate(data_iter):
        data = data.as_in_context(ctx)
        if reshape:
            data[:] = data.reshape((-1, num_inputs))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        #num += nd.sum(predictions == label)
        #den += data.shape[0]
    return acc.get()[1]


def train(train_data, test_data, net, loss_fn, trainer, ctx, one_hot=False, reshape=True, epochs=5):
    losses = []
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            if reshape:
                data[:] = data.reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, 10)       # 10 x inputs 1/0 matrix
            with autograd.record():                     # execute graph
                output = net(data)                      # forward pass
                if one_hot:
                    loss = loss_fn(output, label_one_hot) # calculate loss
                else:
                    loss = loss_fn(output, label)       # calculate loss
            loss.backward()                             # get deriv's (set .grad's)
            trainer()                                   # update weights

            losses.append(nd.mean(loss).asscalar())

        test_accuracy  = evaluate_accuracy(test_data, net, reshape=reshape)
        train_accuracy = evaluate_accuracy(train_data, net, reshape=reshape)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s, Gener_Err %s"
              % (e, nd.mean(loss).asscalar(), train_accuracy, test_accuracy,
                 train_accuracy - test_accuracy))
    return losses


print("Training (Manual version)..")
train(train_data, test_data, net, cross_entropy,
      lambda: sgd(params, learning_rate), ctx, one_hot=True)

#print("Training (Gluon version, single layer)..")
#train(train_data, test_data, net2, softmax_cross_entropy,
#      lambda: sgd2.step(batch_size), ctx, one_hot=False)

print("Training (Gluon version, multiple layers)..")
train(train_data, test_data, net3, softmax_cross_entropy,
      lambda: sgd3.step(batch_size), ctx, one_hot=False)

#print("Training (Gluon version, CNN LeNet).. - very slow 40min")
#train(train_data3d, test_data3d, net4, softmax_cross_entropy,
#      lambda: sgd4.step(batch_size), ctx, one_hot=False, reshape=False)

print("Training (Gluon version, CNN LeNet improved).. - very slow xxmin")
train(train_data3d, test_data3d, net5, softmax_cross_entropy,
      lambda: sgd5.step(batch_size), ctx, one_hot=False, reshape=False)


"""
Training (Gluon version, single layer)..
Epoch 0. Loss: 1.46221, Train_acc 0.794183333333, Test_acc 0.8017, Gener_Err -0.00751666666667
Epoch 1. Loss: 1.14104, Train_acc 0.837516666667, Test_acc 0.8403, Gener_Err -0.00278333333333
Epoch 2. Loss: 0.436261, Train_acc 0.856566666667, Test_acc 0.8562, Gener_Err 0.000366666666667
Epoch 3. Loss: 0.377678, Train_acc 0.866816666667, Test_acc 0.8665, Gener_Err 0.000316666666667
Epoch 4. Loss: 1.3708, Train_acc 0.8732, Test_acc 0.8746, Gener_Err -0.0014
Training (Gluon version, multiple layers)..
Epoch 0. Loss: 0.291714, Train_acc 0.94295, Test_acc 0.9431, Gener_Err -0.00015
Epoch 1. Loss: 0.526049, Train_acc 0.960766666667, Test_acc 0.9591, Gener_Err 0.00166666666667
Epoch 2. Loss: 0.0616461, Train_acc 0.968566666667, Test_acc 0.9643, Gener_Err 0.00426666666667
Epoch 3. Loss: 0.0753418, Train_acc 0.973383333333, Test_acc 0.9692, Gener_Err 0.00418333333333
Epoch 4. Loss: 0.0887326, Train_acc 0.975833333333, Test_acc 0.971, Gener_Err 0.00483333333333
Training (Gluon version, CNN LeNet).. - very slow 40min
Epoch 0. Loss: 0.141891, Train_acc 0.978133333333, Test_acc 0.9794, Gener_Err -0.00126666666667
Epoch 1. Loss: 0.0183562, Train_acc 0.984883333333, Test_acc 0.9822, Gener_Err 0.00268333333333
Epoch 2. Loss: 0.00406375, Train_acc 0.990733333333, Test_acc 0.9896, Gener_Err 0.00113333333333
Epoch 3. Loss: 0.00293897, Train_acc 0.99035, Test_acc 0.9874, Gener_Err 0.00295
Epoch 4. Loss: 0.00559553, Train_acc 0.9951, Test_acc 0.99, Gener_Err 0.0051
Training (Gluon version, CNN LeNet improved).. - very slow 40min
Epoch 0. Loss: 0.0267946, Train_acc 0.990516666667, Test_acc 0.9894, Gener_Err 0.00111666666667
Epoch 1. Loss: 0.0686408, Train_acc 0.995366666667, Test_acc 0.9914, Gener_Err 0.00396666666667
Epoch 2. Loss: 0.0108497, Train_acc 0.996683333333, Test_acc 0.9913, Gener_Err 0.00538333333333
Epoch 3. Loss: 0.0483033, Train_acc 0.998516666667, Test_acc 0.9933, Gener_Err 0.00521666666667
Epoch 4. Loss: 0.000635991, Train_acc 0.998783333333, Test_acc 0.9931, Gener_Err 0.00568333333333
"""
