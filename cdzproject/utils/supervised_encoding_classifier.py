from __future__ import division
import numpy as np
import tensorflow.contrib.learn.python.learn as learn
from sklearn import metrics

batch_size = 32


def get_classification_score(train_encodings, train_labels, test_encodings, test_labels, steps):
    feature_columns = learn.infer_real_valued_columns_from_input(train_encodings)
    classifier = learn.DNNClassifier(hidden_units=[32], n_classes=10, feature_columns=feature_columns)
    classifier.fit(train_encodings, train_labels, steps=steps, batch_size=batch_size)

    # For measuring accuracy
    test_predictions = list(classifier.predict(test_encodings, as_iterable=True))
    return metrics.accuracy_score(test_labels, test_predictions)


# ========== FSDD  ===========
train_encodings = np.load(open('../data/fsdd_train_encodings_2.npy', 'r'))
train_labels = np.load(open('../data/fsdd_train_encodings_2_labels.npy', 'r'))
test_encodings = np.load(open('../data/fsdd_test_encodings_2.npy', 'r'))
test_labels = np.load(open('../data/fsdd_test_encodings_2_labels.npy', 'r'))
train_labels = np.array([np.int32(label) for label in train_labels])
test_labels = np.array([np.int32(label) for label in test_labels])
steps = len(train_encodings) * 200 / batch_size  # ~200 epochs
print 'FSDD accuracy:', get_classification_score(train_encodings, train_labels, test_encodings, test_labels, steps)


# ========== MNIST  ===========
train_encodings = np.load(open('../data/mnist_train_encodings_5.npy', 'r'))
train_labels = np.load(open('../data/mnist_train_encodings_5_labels.npy', 'r'))
test_encodings = np.load(open('../data/mnist_test_encodings_5.npy', 'r'))
test_labels = np.load(open('../data/mnist_test_encodings_5_labels.npy', 'r'))
train_labels = np.array([np.int32(label) for label in train_labels])
test_labels = np.array([np.int32(label) for label in test_labels])
steps = len(train_encodings) * 50 / batch_size  # ~ 50 epochs, fewer epochs are needed because its a larger dataset
print 'MNIST accuracy:', get_classification_score(train_encodings, train_labels, test_encodings, test_labels, steps)
