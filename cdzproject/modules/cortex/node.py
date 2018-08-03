from __future__ import division

import numpy as np
from cdzproject.utils import utils
from cdzproject import db, config


class Node(object):
    """A a node similar to Growing Neural Gas node"""

    def __init__(self, cortex, initial_position, name=None):
        """

        :param cortex:
        :param initial_position:
        :param name:
        :return:
        """
        self.name = utils.name_generator(cortex, name)
        self.cortex = cortex
        self.created_at = self.cortex.timestep

        self.position = initial_position
        self.position_momentum = 0
        self.qty_feedback_packets = 0  # Also equivalent to the number of times it fired (for now)
        self.last_utilized = None
        self.last_encoding = None  # The last encoding this node received

    @property
    def age(self):
        return self.cortex.timestep - self.created_at

    def receive_feedback_packet(self, packet):
        """Receives a packet the gives instructions for how to adjust this nodes connections to its clusters

        This packet is generated from the cluster that recently fired from the other modality. It contains the ID of the
        cluster in this modality that it is most strongly correlated to. This node has its connection with
        packet.cluster strengthened.

        In biological terms this can be thought of the second modality exciting a cluster in this modality and having
        Hebbian learning increasing the connection between the node and the excited cluster.

        :param packet:
        :return:
        """
        amount = packet.strength * config.NODE_TO_CLUSTER_LEARNING_RATE
        db.adjust_node_to_cluster_strength(self, packet.cluster, amount, self.last_encoding)
        self.qty_feedback_packets += 1

    def get_distance(self, position):
        """Returns the euclidean distance between this node and the passed position.

        :param position:
        :return:
        """
        distance_vector = self.position - position
        euclidean_distance = np.linalg.norm(distance_vector)
        return euclidean_distance, distance_vector

    def learn(self, position):
        """ Moves the node in the direction of the passed position.

        :param position:
        :return:
        """
        error, distance_vector = self.get_distance(position)
        self._move_in_direction(-1 * distance_vector)
        self.last_utilized = self.cortex.timestep

    def _move_in_direction(self, direction):
        """Moves the node in the directioan of the passed vector in proportion to the element-wise value.
        IE: the more error in the difference the more we move the node

        :param direction:
        :return:
        """
        self.position += config.NODE_POSITION_LEARNING_RATE * (direction + (config.NODE_POSITION_MOMENTUM_ALPHA * self.position_momentum))
        self.position_momentum = (config.NODE_POSITION_MOMENTUM_DECAY * self.position_momentum) + config.NODE_POSITION_LEARNING_RATE * direction

    def is_underutilized(self):
        """ Returns true if this node is underutilized. This is generally used for the purpose of deleting and
        splitting nodes.

        :return: Boolean
        """
        # We use this trick because `last_utilized` is initially set to None
        time_to_use = max(self.created_at, self.last_utilized)
        return bool((self.cortex.timestep - time_to_use) >= config.NODE_REQUIRED_UTILIZATION)

    def is_new(self):
        if self.last_utilized is None:
            return True

        if self.qty_feedback_packets <= config.NODE_IS_NEW:
            return True

    def uncertainty(self):
        """Returns the uncertainty of this nodes association with its strongest cluster.
        This value is calculated using both self.qty_feedback_packets and self.correlation_variance

        Value is always a number between 0 and 1

        :return:
        """
        # WARNING!
        # If making changes here, you might also want to make changes cluster_correlation.uncertainty()

        # Get the clusters that this node is associated to and their strengths.
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)

        # POSSIBLE IMPROVEMENT: There is much room for improvement here.
        feedback_scale = min(self.qty_feedback_packets / config.NODE_CERTAINTY_AGE_FACTOR, 1)
        certainty = max(strengths)**2 * feedback_scale
        assert 0 <= certainty <= 1
        return 1 - certainty

    def certainty(self):
        return 1 - self.uncertainty()

    def correlation_variance(self):
        """ Calculates the cluster correlation variance. IE: how peaky the cluster connection distribution is.
        A peaky distribution is ideal.
        :return:
        """
        # POSSIBLE IMPROVEMENT: There is much room for improvement here.
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)

        # the distribution of clusters in this cortex that this node probabilistically belongs to.
        # chooses the non-max value
        return 1 - max(strengths)

    def teardown(self):
        db.delete_node(self)

    def get_strongest_cluster(self):
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)
        return clusters[np.argmax(strengths)]
