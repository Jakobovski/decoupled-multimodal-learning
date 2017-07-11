from __future__ import division
from cdzproject.modules.cortex.node_manager import NodeManager
from cdzproject import db


class Cortex(object):

    def __init__(self, brain, name, autoencoder):
        self.name = name
        self.autoencoder = autoencoder
        self.brain = brain

        self.node_manager = NodeManager(self)
        db.node_manager_to_nodes.add(self.node_manager, [], [])

    @property
    def cdz(self):
        return self.brain.cdz

    @property
    def timestep(self):
        return self.brain.timestep

    def receive_sensory_input(self, data, learn=True):
        """Sensory information is passed to the cortex through this function.

        In order to monitor the output of the passed data take a look at brain.output_stream.
        Don't look at the output of this function
        """
        encoding = self.autoencoder.get_encoding(data)
        strongest_cluster = self.node_manager.receive_encoding(encoding, learn=learn)
        return strongest_cluster

    def cleanup(self, delete_new_items=False):
        """Performs maintenance. Does not check for correct timestep."""
        self.node_manager.cleanup(delete_new_items=delete_new_items)

    def create_new_nodes(self):
        """Creates new nodes if needed."""
        self.node_manager.create_new_nodes()
