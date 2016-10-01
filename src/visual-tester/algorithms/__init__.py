# Algorithms ready to test
# Create classes inheriting Algorithm in separate files,
# import instances in main.py


class Algorithm(object):
    # METHODS TO IMPLEMENT
    def __init__(self, **hyperparas):
        pass

    def train(self):
        """Train the algorithm"""
        raise NotImplemented

    def classify(self, data):
        """Classify data"""
        raise NotImplemented

