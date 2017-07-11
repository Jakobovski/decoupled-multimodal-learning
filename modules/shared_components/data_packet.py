from __future__ import division


class DataPacket(object):

    def __init__(self, source_cluster, strength, time, source_node):
        """An encapsulation of information that is sent between parts of the system."""
        self.cluster = source_cluster
        self.strength = strength
        self.time = time
        self.source_node = source_node

    @property
    def cortex(self):
        return self.cluster.cortex
