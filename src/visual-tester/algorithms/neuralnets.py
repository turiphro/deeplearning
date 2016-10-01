# Neural networks as written for the book "Neural networks and Deep Learning" by Nielson

import os
import sys

from algorithms import Algorithm

NNDL_PATH = os.environ.get('NNDL_PATH', '../../../neural-networks-and-deep-learning/src')
sys.path.append(NNDL_PATH)

from mnist_loader import load_data_wrapper
from network import Network


class NeuralNet1(Algorithm):
    def __init__(self, **hyperparas):
        self.hyperparas = hyperparas
        sizes = self.hyperparas.get('sizes', [784, 10])
        self.model = Network(sizes)

    def train(self):
        # get hyper parameters
        epochs = self.hyperparas.get('epochs', 30)
        batch_size = self.hyperparas.get('batch_size', 10)
        eta = self.hyperparas.get('eta', 1.0)
        # get data
        cwd = os.getcwd()
        os.chdir(NNDL_PATH)
        training_data, validation_data, test_data = load_data_wrapper()
        os.chdir(cwd)
        return self.model.SGD(training_data, epochs, batch_size, eta, test_data)

    def classify(self, data):
        return self.model.feedforward(data)

