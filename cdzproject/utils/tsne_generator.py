import numpy as np
from sklearn.manifold import TSNE
from pylab import rcParams
import matplotlib.pyplot as plt


def plot_figure(encodings, labels, save_path):
    rcParams['figure.figsize'] = 5, 5
    plt.scatter(encodings[:, 0], encodings[:, 1], c=labels)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_ticks([])
    frame1.axes.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.savefig(save_path + '_tsne.png', pad_inches=0)
    # plt.show()


def generate_tsne(encodings_path, labels_path):
    encodings = np.load(open(encodings_path, 'r'))
    labels = np.load(open(labels_path, 'r'))
    tsne_model = TSNE(verbose=4)

    # Only draw a subsets as we don't want to crowd the image
    encodings_subset = encodings[0:min(len(encodings), 15000)]
    labels_subset = labels[0:min(len(encodings), 15000)]

    tsne_results = tsne_model.fit_transform(encodings_subset)
    plot_figure(tsne_results, labels_subset, encodings_path)


if __name__ == '__main__':
    generate_tsne('../data/mnist_train_encodings.npy', '../data/mnist_train_encodings_labels.npy')
    # generate_tsne('../data/fsdd_train_encodings.npy', '../data/fsdd_train_encodings_labels.npy')
