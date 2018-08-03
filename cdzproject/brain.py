from collections import deque

from cdzproject import config, db
from cdzproject.modules.cortex.cortex import Cortex
from cdzproject.modules.cdz.cdz import CDZ


class Brain:

    def __init__(self):
        """This is the entry point and root object in the architecture."""
        self.timestep = 0
        self.cortices = {}
        self.cdz = CDZ(self)
        self.output_stream = deque(maxlen=10)

    def add_cortex(self, cortex_name, autoencoder):
        """Initializes and adds a cortex by the given name to the brain instance.

        Args:
            cortex_name (string): the name of the cortex (ex: audio, visual)
            autoencoder (Autoencoder): the cortex's autoencoder
        """
        if self.cortices.get(cortex_name):
            raise Exception('A cortex by this name is already present in this brain.')

        new_cortex = Cortex(self, cortex_name, autoencoder)
        self.cortices[cortex_name] = new_cortex
        return new_cortex

    def get_cortex(self, cortex_name):
        """Gets a cortex by name

        Args:
            cortex_name (string):

        Returns:
            Cortex: the requested cortex
        """
        return self.cortices[cortex_name]

    def increment_timestep(self, amount=1):
        """Increments the brain's timestep.

        Args:
            amount (int, optional): The amount to increment the timestep by. Defaults to 1.
        """
        self.timestep += amount

    def recieve_sensory_input(self, cortex, data, learn=True):
        """Takes sensory data and sends it to the cortex specified

        Args:
            cortex (Cortex):  a cortex object.
            data (nparray): sensory data
            learn (bool, optional): Parameter specifying if the cortex should learn, or just pass through
        """
        cortex.receive_sensory_input(data, learn=learn)

    def cleanup(self, force=False, delete_new_items=False):
        """Performs maintenance. Deletes unused nodes/clusters.
        Only runs if the timestep is a multiple of config.BRN_CLEANUP_FREQUENCY , or force=True

        Args:
            force (bool, False): Default to false
            delete_new_items (bool, False): determines if we should delete newly created nodes/clusters. This is useful
                                            for performing measurements at the end of learning.
        """
        if force or delete_new_items or self.timestep % config.BRN_CLEANUP_FREQUENCY == 0:
            print "====== Start cleanup ====="
            for cortex in self.cortices.values():
                cortex.cleanup(delete_new_items=delete_new_items)

            db.cleanup()
            self.build_nrnd_indexes(force=True)
            print "====== End cleanup ======"

    def create_new_nodes(self):
        """Creates new nodes in each cortex as needed."""
        if self.timestep % config.BRN_NEURAL_GROWTH_FREQUENCY == 0:
            print "====== Start neural growth ====="
            for cortex in self.cortices.values():
                cortex.create_new_nodes()
            print "====== End neural growth ======="

    def build_nrnd_indexes(self, force=False):
        """Builds the nearest node index. This is used to increase the performance of the algorithm.
        This only runs if force=True or if the timestep is a multiple of config.NRND_BUILD_FREQUENCY
        Args:
            force (bool, False):
        """
        if force or self.timestep % config.NRND_BUILD_FREQUENCY == 0:
            for cortex in self.cortices.values():
                cortex.node_manager.build_nrnd_index()
