from __future__ import division
from collections import defaultdict

from cdzproject import config


class ClusterCorrelation(object):

    def __init__(self, cluster, cdz):
        """This object represents a relationship between one cluster and other clusters in the CDZ"""
        self.cdz = cdz
        self.cluster = cluster
        self.age = 1
        self.connections = defaultdict(int)
        self.cluster_objects = {}
        # A list of clusters that reference this cluster
        self.ref_clusters = []

    def update(self, q_packet, new_packet):
        # Make sure we are using the right connection obj/packet
        assert q_packet.cluster == self.cluster

        # Don't correlate the cortex to itself
        assert q_packet.cortex != new_packet.cortex

        # Because of the normalization we don't want values too big. So lets just limit the values to 1
        assert max(q_packet.strength, new_packet.strength) <= 1

        # Calculate the amount to weigh the q_packet... older packets are weighed less.
        time_diff = (new_packet.time - q_packet.time)
        assert time_diff >= 0

        temporal_weight = self.cdz.GAUSSIAN[time_diff]
        correlation_update = self.cdz.LEARNING_RATE * temporal_weight * new_packet.strength * q_packet.strength

        # Increase the connection strength between the new packet and the existing(remaining)
        # packets in proportion to their Gaussian overlap and their classification certainty
        self.connections[new_packet.cluster.name] += correlation_update
        self._normalize()
        self.age += 1

        # Store a reference to the cluster object
        self.cluster_objects[new_packet.cluster.name] = new_packet.cluster

    def _normalize(self, dict_to_normalize=None):
        """Normalizes the connections so that they sum to one.
        This ensures competition between the connections.
        """
        if not dict_to_normalize:
            dict_to_normalize = self.connections

        val_sum = sum(dict_to_normalize.values())
        for key, val in dict_to_normalize.iteritems():
            dict_to_normalize[key] = val / val_sum

        return dict_to_normalize

    def remove_cluster(self, cluster):
        """Removes a cluster from this correlation"""
        del self.connections[cluster.name]
        del self.cluster_objects[cluster.name]
        self._normalize()

    def add_ref(self, cluster):
        if cluster not in self.ref_clusters:
            self.ref_clusters.append(cluster)

    def get_strongest_correlation(self):
        """Returns the name of the strongest cluster that this cluster is correlated to."""
        cluster_name = max(self.connections, key=lambda conn_name: self.connections[conn_name])
        cluster = self.cluster_objects[cluster_name]
        strength = self.connections[cluster_name]
        return cluster, strength

    def uncertainty(self):
        """ Returns the certainty that this cluster is associated to its strongest cluster."""
        # POSSIBLE IMPROVEMENT: There is much room for improvement here.

        # WARNING!
        # If making changes here, you might also want to make changes in node.uncertainty()
        cluster, strength = self.get_strongest_correlation()

        feedback_scale = min(self.age / config.CE_CERTAINTY_AGE_FACTOR, 1)
        certainty = strength**2 * feedback_scale
        assert 0 <= certainty <= 1
        return 1 - certainty

    def certainty(self):
        return 1 - self.uncertainty()
