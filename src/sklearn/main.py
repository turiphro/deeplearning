import random
from sklearn import datasets, metrics, svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def function_approx():
    # init
    clf = MLPClassifier(solver='sgd', alpha=1e-5,
                        activation='relu', hidden_layer_sizes=(10),
                        learning_rate='constant', learning_rate_init=0.001,
                        random_state=1, early_stopping=False,
                        verbose=True)

    def fn(x, y):
        return round(x + y)

    # train
    _MAX = 3
    X = []
    y = []
    for i in range(1000):
        _x, _y = random.randint(0, _MAX), random.randint(0, _MAX)
        #_xnoise, _ynoise = random.random(), random.random()
        _xnoise, _ynoise = 0, 0
        X.append([_x / _MAX + _xnoise, _y / _MAX + _ynoise])
        y.append(fn(_x, _y))

    print(X)
    print(y)
    clf.fit(X, y)
    print("weights:", clf.coefs_)
    print("biases: ", clf.intercepts_)

    # classify
    for i in range(10):
        _x, _y = random.uniform(0, _MAX), random.uniform(0, _MAX)
        classification = clf.predict([[_x / _MAX, _y / _MAX]])
        print("Classified {} as {} (should be {})".format(
            [_x, _y], classification, fn(_x, _y)))


def mnist():
    #digits = datasets.load_digits() # subsampled version
    mnist = datasets.fetch_mldata("MNIST original")
    print("Got the data.")
    X, y = mnist.data / 255., mnist.target
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    #images_and_labels = list(zip(digits.images, digits.target))
    #for index, (image, label) in enumerate(images_and_labels[:4]):
    #    plt.subplot(2, 4, index + 1)
    #    plt.axis('off')
    #    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #    plt.title('Training: %i' % label)

    classifiers = [
        #("SVM", svm.SVC(gamma=0.001)), # TODO doesn't finish; needs downsampled version?
        ("NN", MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                             solver='sgd', verbose=10, tol=1e-4, random_state=1,
                             learning_rate_init=.1)),
    ]

    for name, classifier in classifiers:
        print(name)
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)

        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(y_test, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))

        #images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
        #for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        #    plt.subplot(2, 4, index + 5)
        #    plt.axis('off')
        #    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        #    plt.title('Prediction: %i' % prediction)

        #plt.show()


if __name__ == '__main__':
    function_approx()
    mnist()
