#!/usr/bin/env python
# Visual tester for machine learning algorithms

import sys

import cv2
import numpy as np

from algorithms.neuralnets import NeuralNet1

CONTINUOUS = True # classify continuously


def cli_choose(options):
    choice = None
    while choice is None:
        print("Choose:")
        for i, option in enumerate(options):
            print("{}: {}".format(i, option))
        try:
            choice = int(input(">> "))
            assert 0 <= choice < len(options)
        except Exception:
            choice = None # try again
    return choice


def squarify(x, y, w, h):
    """Crop to square (centre of window)"""
    if w > h:
        x += int((w - h) / 2)
        w = h
    elif h > w:
        y += int((h - w) / 2)
        h = w
    return x, y, w, h


def cut_square(frame):
    x, y, w, h = squarify(0, 0, frame.shape[0], frame.shape[1])
    return frame[x:x+w, y:y+h, :]


if __name__ == '__main__':
    algorithms = [
        ("Neural net 1", NeuralNet1,
            {'sizes': [784, 30, 10], 'epochs': 3, 'batch_size': 10, 'eta': 3.0}),
    ]

    # choose classifier
    choice = cli_choose([descr for descr, _, _ in algorithms])
    descr, klass, hyperparas = algorithms[choice]
    print("Running algorithm: {}".format(descr))

    # create and train
    print("Training:")
    algorithm = klass(**hyperparas)
    algorithm.train()

    print("Entering classification loop.")
    WEBCAM_DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(WEBCAM_DEVICE)
    while True:
        # show webcam stream until hitting a key:
        frame = None
        last_guess = ""
        while cv2.waitKey(1) == -1:
            ret, frame = cap.read()
            frame = cut_square(frame)
            cv2.imshow("Webcam", frame)
            if CONTINUOUS:
                break # first frame is all we need

        # transform image to desired format: binary inverted {0,1} 784x1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # to grey values
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # threshold
                                      cv2.THRESH_BINARY, 201, 25)
        data = cv2.resize(frame, (28, 28), interpolation=cv2.INTER_AREA) # scale down
        #data = np.round(1 - (data / 255.0)) # to {0,1} set, black is background
        data = 1.0 * (data < 230) # threshold (binary input) and invert
        cv2.imshow("Input frame", frame)
        cv2.imshow("Input data",
                   cv2.resize(data, frame.shape[:2], interpolation=cv2.INTER_NEAREST))
        #cv2.waitKey()
        data = np.reshape(data, (784, 1))

        # classify
        classification = algorithm.classify(data)
        max_likelihood = np.argmax(classification)
        print("Classification:")
        for i, neuron in enumerate(classification):
            print("%2d: %2.4f %s" %
                  (i, neuron[0], "<--" if i == max_likelihood else ""))

