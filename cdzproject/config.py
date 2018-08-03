"""A global configuration file

WARNING: Many of these parameters are dependent on each other. If you make adjustments and things break its most likely
because you did not adjust the dependent parameters as well.  

"""
from __future__ import division

EPOCHS = 20

# The size of the subset to use for training. I often keep this small to save time and test new ideas.
# NOTE: If you increase this then you may want to adjust the other parameters accordingly.
TRAINING_SET_SIZE = 55000

# How often the maintenance function is run.
BRN_CLEANUP_FREQUENCY = TRAINING_SET_SIZE * .25
BRN_NEURAL_GROWTH_FREQUENCY = TRAINING_SET_SIZE * .3

# We don't want to create growth too often because it causes lots of noise
assert BRN_CLEANUP_FREQUENCY <= BRN_NEURAL_GROWTH_FREQUENCY

# ======================================================================================
# ===================================== Cluster ========================================
# ======================================================================================
CLUSTER_REQUIRED_UTILIZATION = TRAINING_SET_SIZE * .5
CLUSTER_NODE_LEARNING_RATE = 0.02

# ======================================================================================
# ======================================= Node =========================================
# ======================================================================================

# Determines if it should be deleted and split into new nodes
NODE_REQUIRED_UTILIZATION = TRAINING_SET_SIZE * 1.01

# We want the nodes to be required to be used less frequently than the clusters
assert NODE_REQUIRED_UTILIZATION > CLUSTER_REQUIRED_UTILIZATION

NODE_POSITION_LEARNING_RATE = 0.04
NODE_POSITION_MOMENTUM_DECAY = 0.5
NODE_POSITION_MOMENTUM_ALPHA = 0.0
NODE_TO_CLUSTER_LEARNING_RATE = 0.05

# The number of times node has fired for it to not be classified as new
# Determines if the node should be split into new nodes
# Also effects the uncertainty measurement
NODE_IS_NEW = 25
NODE_CERTAINTY_AGE_FACTOR = NODE_IS_NEW

# ======================================================================================
# ==================================== Node Manager ====================================
# ======================================================================================

# Nearest node optimizer
# improves speed, but reduces accuracy as indexes are rebuilt periodically
# and its approximate. See: https://github.com/spotify/annoy
NRND_OPTIMIZER_ENABLED = True
NRND_BUILD_FREQUENCY = TRAINING_SET_SIZE / 10
NRND_N_TREES = 500
NRND_SEARCH_K = NRND_N_TREES * 5

# The maximum avg momentum of the nodes positions before we using ANNOY optimization. This is not relevant for this
# example but may be relevant for other datasets.
NRND_MAX_AVG_DISTANCE_MOMENTUM = 1e50
AVG_DISTANCE_MOMENTUM_DECAY = 0.5

# Maximum number of nodes allowed in each node
# manager (one NM per cortex)
MAX_NODES = 6000
INITIAL_NODES = 250
assert INITIAL_NODES <= MAX_NODES

# The maximum amount of correlation variance a node is allowed to have for it not to be broken up
NODE_SPLIT_MAX_CORRELATION_VARIANCE = 5e-3
NODE_SPLIT_MAX_QTY = max(TRAINING_SET_SIZE / 1000, 5)


# ======================================================================================
# ============================== Correlation Engine (CDZ) ==============================
# ======================================================================================
CE_LEARNING_RATE = 0.02
# Overrides the Gaussian distribution and just uses a [1,0,0,....] distribution
# Meaning only two events that happen at exactly the same time are correlated
CE_IGNORE_GAUSSIAN = True
CE_CORRELATION_WINDOW_STD = 0.65
CE_CORRELATION_WINDOW_MAX = 10

CE_CERTAINTY_AGE_FACTOR = NODE_CERTAINTY_AGE_FACTOR
