from __future__ import division
from annoy import AnnoyIndex

from cdzproject.modules.cortex.node import Node
from cdzproject import db, config
from cdzproject.utils import utils
from cdzproject.modules.cortex.cluster import Cluster


class NodeManager(object):

    def __init__(self, cortex, name=None):
        """ A node manger is tool to manage the the nodes.

        Functionality:
            - deletes old nodes
            - adds new nodes
            - clusters existing nodes
            - receives encodings from autoencoder
            - passes encodings up to the autoencoder (generative)
            - passes clusters down to the CDZ
        """

        # A list of all the clusters contained in this cortex
        self.cortex = cortex
        self.name = utils.name_generator(cortex, name)
        self.last_fired_node = None
        self.finished_initial = False
        self.nn_index = None

        self.avg_distance = 0
        self.distance_count = 0
        self.avg_distance_momentum = 0

    @property
    def nodes(self):
        return db.get_node_managers_nodes(self)

    def build_nrnd_index(self):
        """Builds/rebuilds an index for finding the nearest nodes. This is for improving performance."""
        if config.NRND_OPTIMIZER_ENABLED:
            if self.nn_index or (self.finished_initial and abs(self.avg_distance_momentum) < config.NRND_MAX_AVG_DISTANCE_MOMENTUM and self.distance_count > 1000):
                print "Building nearest node index..."
                dimensions = len(self.nodes[0].position)
                self.nn_index = AnnoyIndex(dimensions, metric='euclidean')

                idx = 0
                for node in self.nodes:
                    assert node.position is not None
                    self.nn_index.add_item(idx, node.position)
                    idx += 1

                self.nn_index.build(config.NRND_N_TREES)

    def _update_avg_distance(self, distance):
        """Keeps a moving average of node distances."""
        self.distance_count += 1
        new_avg = self.avg_distance + (distance - self.avg_distance) / self.distance_count
        self.avg_distance_momentum = (config.AVG_DISTANCE_MOMENTUM_DECAY * self.avg_distance_momentum) + new_avg - self.avg_distance

    def receive_encoding(self, encoding, learn=True):
        # Initialize nodes if they have not yet been initialized
        self._add_initial_nodes(encoding)

        # Find nearest node to encoding
        nearest_node, distance = self._find_nearest_node(encoding)
        nearest_node.last_encoding = encoding

        # Find nearest_node's strongest cluster
        strongest_cluster = nearest_node.get_strongest_cluster()
        self.last_fired_node = nearest_node

        # Move the node towards the encoding
        if learn:
            nearest_node.learn(encoding)
            self._update_avg_distance(distance)

        # Fire the cluster so that it sends a packet to cdz
        # POSSIBLE IMPROVEMENT: strength can be a function of distance
        strength = 1
        strongest_cluster.excite_cdz(strength, nearest_node, learn=learn)
        return strongest_cluster

    def receive_feedback_packet(self, packet):
        """Receives a packet the gives instructions for how to adjust the most recent nodes connections to it's internal
         clusters.

        This packet is generated from the cluster that recently fired from the other modality. It contains the ID of the
        cluster in this modality that it is most strongly correlated to. The node in this modality has its connection
        with packet.cluster strengthened.

        In biological terms this can be thought of the second modality exciting a cluster in this modality and having
        Hebbian learning increasing the connection between the node and the excited cluster.
        """

        # Find the node that most recently fired
        # Increase its connection strength to packet.cluster
        self.last_fired_node.receive_feedback_packet(packet)

    def reconstruct(self, packet):
        """ Given a cluster this function finds it encoding and reconstructs it through the autoencoder."""
        # TODO
        raise NotImplementedError()

    def cleanup(self, delete_new_items=False):
        """This function does a number of required maintenance procedures.
            - deletes underutilized nodes/clusters
            - deleted new nodes/clusters (used for measuring score after training)
        """
        self._delete_underutilized_items()

        if delete_new_items:
            self._delete_new_items()

    def create_new_nodes(self):
        """Creates new nodes in locations that have a high variance."""
        if len(self.nodes) >= config.MAX_NODES:
            return

        num_nodes_added = 0

        # Sort the nodes in this cortex by their correlation_variance.
        sorted_nodes = sorted(self.nodes, key=(lambda node: node.correlation_variance()), reverse=True)
        assert sorted_nodes[0].correlation_variance() >= sorted_nodes[-1].correlation_variance()

        def is_eligible(node):
            """ Determines if a node is eligible for splitting.
                - not new
                - not underutilized
                - has high variance in its connections to clusters
            """
            has_high_variance = bool(node.correlation_variance() > config.NODE_SPLIT_MAX_CORRELATION_VARIANCE)
            return has_high_variance and not node.is_new() and not node.is_underutilized()

        # Loop through the nodes and create new ones nearby the ones that are ambiguous
        for node in sorted_nodes:

            if not is_eligible(node):
                continue

            # Get all the nodes clusters
            clusters, strengths, positions, counts = db.get_nodes_clusters(node, include_all=True)

            assert len(clusters) == len(strengths)
            assert len(clusters) == len(positions)

            for idx, cluster in enumerate(clusters):
                if counts[idx] > 3:
                    new_position = positions[idx]
                    new_node = Node(self.cortex, new_position)
                    new_cluster = Cluster(self.cortex, 'cluster_' + new_node.name)
                    db.add_node(new_node, new_cluster)
                    num_nodes_added += 1

            new_node = Node(self.cortex, node.position)
            new_cluster = Cluster(self.cortex, 'cluster_' + new_node.name)
            db.add_node(new_node, new_cluster)
            num_nodes_added += 1

            node.teardown()
            num_nodes_added -= 1

            if len(self.nodes) >= config.MAX_NODES or num_nodes_added >= config.NODE_SPLIT_MAX_QTY:
                break

        self.build_nrnd_index()

    def _delete_underutilized_items(self):
        """Removes underutilized nodes/clusters from this node manager and the DB."""
        # Need to make copy of list or else have issues!
        for node in self.nodes[:]:
            if node.is_underutilized():
                node.teardown()

    def _delete_new_items(self):
        """Removes new nodes/clusters from this node manager and the DB."""
        # We make a copy of list or else have issues!
        for node in self.nodes[:]:
            if node.is_new():
                node.teardown()

    def _add_initial_nodes(self, encoding):
        """ Adds nodes near the passed encoding if all the initial nodes have not been initialized yet."""
        if len(self.nodes) <= config.MAX_NODES and not self.finished_initial:
            node = Node(self.cortex, encoding)
            cluster = Cluster(node.cortex, 'cluster_' + node.name)
            db.add_node(node, cluster, initial=True)

        if len(self.nodes) >= config.INITIAL_NODES:
            self.finished_initial = True

    def _find_nearest_node(self, encoding):
        """ Returns the node that is nearest to the passed encoding"""
        if config.NRND_OPTIMIZER_ENABLED and self.nn_index:
            if config.NRND_SEARCH_K:
                nn_idxs, distances = self.nn_index.get_nns_by_vector(encoding, 1, include_distances=True, search_k=config.NRND_SEARCH_K)
            else:
                nn_idxs, distances = self.nn_index.get_nns_by_vector(encoding, 1, include_distances=True)
            return self.nodes[nn_idxs[0]], distances[0]
        else:
            tuples = [(node, node.get_distance(encoding)[0]) for node in self.nodes]
            assert len(tuples) > 0
            return min(tuples, key=lambda x: x[1])
