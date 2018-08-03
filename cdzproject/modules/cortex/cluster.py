import numpy as np

from cdzproject.utils import utils
from cdzproject import db, config
from cdzproject.modules.shared_components.data_packet import DataPacket


class Cluster(object):

    def __init__(self, cortex, name, required_utilization=config.CLUSTER_REQUIRED_UTILIZATION):
        self.name = utils.name_generator(cortex, name)
        self.cortex = cortex
        self.created_at = self.cortex.timestep
        self.last_fired = None
        self.last_feedback_packet = None
        self.REQUIRED_UTILIZATION = required_utilization

    @property
    def age(self):
        return self.timestep - self.created_at

    @property
    def nodes(self):
        return db.get_clusters_nodes(self)

    @property
    def cdz(self):
        return self.cortex.brain.cdz

    @property
    def node_manager(self):
        return self.cortex.node_manager

    @property
    def timestep(self):
        return self.cortex.brain.timestep

    def excite_cdz(self, strength, source_node, learn=True):
        """ Sends a packet to the cdz. This happens when this cluster is excited."""
        packet = DataPacket(self, strength, self.timestep, source_node)
        self.last_fired = self.cortex.timestep

        # We update the relationship between this cluster and the node that fired
        if learn:
            amount = config.CLUSTER_NODE_LEARNING_RATE
            db.adjust_cluster_to_node_strength(self, source_node, amount)

        self.cdz.receive_packet(packet, learn=learn)

    def is_underutilized(self):
        time_to_use = max(self.created_at, self.last_fired, self.last_feedback_packet)
        return bool((self.cortex.timestep - time_to_use) >= self.REQUIRED_UTILIZATION)

    def receive_feedback_packet(self, feedback_packet):
        self.last_feedback_packet = self.cortex.timestep
        self.node_manager.receive_feedback_packet(feedback_packet)

    def get_strongest_node(self):
        """ Returns the node that this cluster has the strongest correlation to.

        If we assume certain statistical properties about the dataset then this node will approximately be the average
        representation of this cluster.

        Return:
            The node that this cluster is most strongly associated to.
        """
        nodes, strengths = db.get_clusters_nodes(self, include_strengths=True)

        if len(nodes) == 0:
            return None

        return nodes[np.argmax(strengths)]
