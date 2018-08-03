from __future__ import division
import pprint

from cdzproject.utils import utils
from cdzproject import db, config
from cdzproject.brain import Brain
from cdzproject.modules.cortex.autoencoder import Autoencoder

# == CHOOSE ONE OF THE DATASETS BELOW ==
# - Different datasets demonstrate that the algorithm work on a variety of data distributions
#   and is not fine-tuned for MNIST+FSDD.
# from cdzproject.utils import mnist_1d_dataset as dataset   # (simulated audio encodings, with ints)
# from cdzproject.utils import mnist_mnist_dataset as dataset   # (trains MNIST against MNIST 5 digits vs. 5 digits)
from cdzproject.utils import encodings_mnist_fsdd as dataset  # (real audio encodings)

# Print the configuration info so we can keep track of what the configuration was for the given run
pprint.pprint(config.__dict__)

brain = Brain()
visual_autoencoder = Autoencoder()
audio_autoencoder = Autoencoder()
visual_cortex = brain.add_cortex('visual', visual_autoencoder)
audio_cortex = brain.add_cortex('audio', audio_autoencoder)

NUM_EXAMPLES = config.EPOCHS * config.TRAINING_SET_SIZE

for timestep in range(NUM_EXAMPLES):
    brain.increment_timestep()

    # Get random visual and audio encodings
    visual_input, audio_input, class_label = dataset.get_random_train_data()

    # Send the encodings to the brain
    brain.recieve_sensory_input(visual_cortex, visual_input)
    brain.recieve_sensory_input(audio_cortex, audio_input)

    brain.cleanup()       # Deletes under-utilized nodes/clusters
    brain.build_nrnd_indexes()  # Rebuild NearestNode indexes (For performance onlt)
    utils.print_info(dataset, brain, NUM_EXAMPLES)

    # Stop creating new nodes towards the end of training
    # This is mostly so the new nodes don't interfere with the score as they are not fully trained yet
    # And because our scoring algorithm is not sophisticated
    if timestep / NUM_EXAMPLES < .75:
        brain.create_new_nodes()


# == Finally ==
brain.cleanup(force=True)
db.verify_data_integrity()

utils.print_info(dataset, brain, NUM_EXAMPLES)
utils.print_score(dataset, brain)
