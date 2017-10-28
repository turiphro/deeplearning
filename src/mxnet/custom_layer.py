import mxnet as mx
from mxnet import NDArray as nd
from mxnet.gluon import Block


def relu(X):
    return nd.maximum(X, 0)


class MyDense(Block):

    def __init__(self, units, in_units=0, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.units = units
            # gluon.Parameter with 0-valued shape elements means: will be filled in later
            self._in_units = in_units
            # Add parameters to internal ParameterDict, indicating the desired shape
            self.weight = self.params.get(
                'weight', init=mx.init.Xavier(magnitude=2.24),
                shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        # Gluon / NDArray can create the backward pass (derivative) automatically
        # based on the forward definition (will be called within .record() block)
        with x.context:
            linear = nd.dot(x, self.weight.data()) + self.bias.data()
            activation = relu(linear)
            return activation
