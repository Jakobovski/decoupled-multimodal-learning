""" A data set that consists of 5 digits of MNIST as visual and different 5 digits as audio.

It trains the network to classify the digits {0,1,2,3,4} in terms of {5,6,7,8,9} and vice versa.
"""
from __future__ import division
from collections import defaultdict
import random
import numpy as np

# Load visual data
v_train_data = np.load('data/mnist_train_encodings_3.npy')
v_train_labels = np.load('data/mnist_train_encodings_3_labels.npy')

v_test_data = np.load('data/mnist_test_encodings_3.npy')
v_test_labels = np.load('data/mnist_test_encodings_3_labels.npy')

data_dict = defaultdict(list)
for idx, label in enumerate(v_train_labels):
    data_dict[label].append(v_train_data[idx])


test_data_dict = defaultdict(list)
for idx, label in enumerate(v_test_labels):
    test_data_dict[label].append(v_test_data[idx])


# Fix the train labels
v_train_data = [data_dict[label] for label in range(5)]
v_train_labels = [[label for _ in data_dict[label]]for label in range(5)]
v_train_labels = np.ravel(v_train_labels)  # Flattens the list

# Fix the test labels
v_test_data = [test_data_dict[label] for label in range(5)]
v_test_labels = [[label for _ in test_data_dict[label]]for label in range(5)]
v_test_labels = np.ravel(v_test_labels)  # Flattens the list


# Fix the train labels
a_train_data = [data_dict[label] for label in range(5, 10)]
a_train_labels = [[label for _ in data_dict[label]]for label in range(5, 10)]
a_train_labels = np.ravel(a_train_labels)  # Flattens the list

# Fix the test labels
a_test_data = [test_data_dict[label] for label in range(5, 10)]
a_test_labels = [[label for _ in test_data_dict[label]]for label in range(5, 10)]
a_test_labels = np.ravel(a_test_labels)  # Flattens the list


def get_random_train_data():
    v_label = random.randint(0, 5)
    data_dict[v_label]
    visual_encoding = np.random.choice(data_dict[v_label])
    audio_encoding = np.random.choice(data_dict[v_label + 5])
    return visual_encoding, audio_encoding, np.float32(v_label)
