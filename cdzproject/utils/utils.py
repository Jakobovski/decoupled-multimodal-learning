from __future__ import division
from datetime import datetime
from collections import defaultdict
import math

from cdzproject import db, config

counter = []


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def name_generator(cortex, name=None):
    """Returns a name as a string.
    If no name is given then the name will be a concatenation of the passed `cortex.name`
    and a 9 digit random number (Ex: visual_012345678).

    If a name is passed then the passed name will be returned.
    """
    counter.append(0)
    if name is None:
        return str(cortex.name + '_' + str(len(counter)))  # + '_' + str(cortex.timestep))
    else:
        return str(name)


def _get_score(encodings, labels, cortex):
    labels_dict = defaultdict(list)
    scores = []
    strongest = []

    def most_common(lst):
        """ Returns the most common value in a list"""
        return max(set(lst), key=lst.count)

    for idx, data in enumerate(encodings):
        excited_cluster = cortex.receive_sensory_input(data, learn=False)

        # Ignore new clusters as they are not fully trained.
        if cortex.cdz.correlations.get(excited_cluster.name) and not excited_cluster.is_underutilized():
            other_modality_cluster = cortex.cdz.correlations[excited_cluster.name].get_strongest_correlation()[0]
            labels_dict[labels[idx]].append(other_modality_cluster.name)

    print '----------'
    print "Cortex:", str(cortex.name)

    for label_name, array in labels_dict.iteritems():
        mst_common_cluster_name = most_common(array)
        strongest.append(mst_common_cluster_name)
        percent = array.count(mst_common_cluster_name) / len(array)
        scores.append(percent)
        print 'Label', label_name, ' -- ', percent, '--', mst_common_cluster_name

    print "Avg score:", sum(scores) / len(scores)
    print "# Unique Top Cluster:", len(set(strongest))


def print_score(dataset, brain):
    """ Calculates scores of the system.  Calculates both training and test in both
        modalities.

        WARNING: This function ignores new clusters as they may not be trained and should not count.
    """
    print "======= Training Score ======="
    _get_score(dataset.v_train_data, dataset.v_train_labels, brain.get_cortex('visual'))
    _get_score(dataset.a_train_data, dataset.a_train_labels, brain.get_cortex('audio'))

    print "======= Test Score ==========="
    _get_score(dataset.v_test_data, dataset.v_test_labels, brain.get_cortex('visual'))
    _get_score(dataset.a_test_data, dataset.a_test_labels, brain.get_cortex('audio'))


def print_info(dataset, brain, num_runs):
    timestep = brain.timestep

    if timestep % 200 == 0:
        print timestep, (str(int(timestep * 100 / num_runs))) + '%', "{0:.2f}".format(timestep / config.TRAINING_SET_SIZE)

    if timestep and timestep % config.TRAINING_SET_SIZE == 0:
        strongest_audio_clusters = defaultdict(list)
        strongest_visual_clusters = defaultdict(list)

        # Get the strongest clusters
        for node_name, data in db.nodes_to_clusters.data.iteritems():
            node = data['obj']
            cluster = node.get_strongest_cluster()
            if node.correlation_variance() <= 0.05 and not node.is_new():
                if 'audio' in node.name:
                    strongest_audio_clusters[cluster.name].append(node)
                elif 'visual' in node.name:
                    strongest_visual_clusters[cluster.name].append(node)

        print '================ System Info ================='
        # for node_name, data in db.nodes_to_clusters.data.iteritems():
        #     node = data['obj']
        #     if 'visual' in node_name:
        #         print 'Node:', '-', node_name, '-', '-', node.get_strongest_cluster().name, node.age, '-', node.qty_feedback_packets, '-', node.correlation_variance()
        # print '====='
        # for node_name, data in db.nodes_to_clusters.data.iteritems():
        #     node = data['obj']
        #     if 'audio' in node_name:
        #         print 'Node:', '-', node_name, '-', '-', node.get_strongest_cluster().name, node.age, '-', node.qty_feedback_packets, '-', node.correlation_variance(), node.position

        # clusters, strengths = db.get_nodes_clusters(node, include_strengths=True)
        # print strengths

        print ''
        print "Audio clusters:", len([key for key in db.clusters_to_nodes.data.keys() if 'audio' in key])
        print "visual clusters:", len([key for key in db.clusters_to_nodes.data.keys() if 'visual' in key])
        print ''
        print "Audio nodes:", len([key for key in db.nodes_to_clusters.data.keys() if 'audio' in key])
        print "visual nodes:", len([key for key in db.nodes_to_clusters.data.keys() if 'visual' in key])
        print ''
        print "=== # Old and low_variance clusters ==="
        print ">> Audio clusters: ", len(strongest_audio_clusters)
        print ">> visual clusters: ", len(strongest_visual_clusters)
        print ''

        print '================ End System Info ==============='
        print_score(dataset, brain)
