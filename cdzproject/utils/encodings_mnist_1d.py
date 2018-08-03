"""This is a dataset where the audio encoding is simply an integer.
This is useful for testing, and playing around.
"""

from __future__ import division
import random
import numpy as np

# Load visual data
v_train_data = np.load('data/mnist_train_encodings.npy')
v_train_labels = np.load('data/mnist_train_encodings_labels.npy')

v_test_data = np.load('data/mnist_test_encodings.npy')
v_test_labels = np.load('data/mnist_test_encodings_labels.npy')

# Load audio data
a_test_data = range(10)
a_train_labels = range(10)

a_test_data = range(10)
a_test_labels = range(10)


def get_random_train_data():
    rand_idx = random.randint(0, len(v_train_labels) - 1)
    visual_encoding = v_train_data[rand_idx]
    label = v_train_labels[rand_idx]
    audio_encoding = [np.float32(label)]
    return visual_encoding, audio_encoding, np.float32(label)
