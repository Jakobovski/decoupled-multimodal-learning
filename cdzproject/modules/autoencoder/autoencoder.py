import numpy as np
import tensorflow as tf
from yadlt.models.autoencoder_models.deep_autoencoder import DeepAutoencoder


class Autoencoder:

    def __init__(self, neurons_per_layer, pretrain, pretrain_epochs, finetune_epochs, finetune_batch_size):
        """A wrapper for a yadlt deep autoencoder."""
        self.autoencoder_config = {
            'layers': neurons_per_layer,
            'enc_act_func': [tf.nn.sigmoid],
            'dec_act_func': [tf.nn.sigmoid],
            'finetune_num_epochs': finetune_epochs,
            'finetune_loss_func': 'mean_squared',
            'finetune_dec_act_func': [tf.nn.sigmoid],
            'finetune_enc_act_func': [tf.nn.sigmoid],
            'finetune_opt': 'adam',
            'finetune_learning_rate': 1e-4,
            'finetune_batch_size': finetune_batch_size,
            'do_pretrain': pretrain,
            'num_epochs': [pretrain_epochs],
            'verbose': 1,
            'corr_frac': [.5],
            'corr_type': ["masking"]
        }

        self.encoder = DeepAutoencoder(**self.autoencoder_config)

    def train(self, data):
        # Normalize the data
        # We need to do this because we are using sigmoid that have a range of [-1, 1]
        data = np.array([blurb / max(abs(blurb)) for blurb in data])
        np.random.shuffle(data)

        if self.autoencoder_config['do_pretrain']:
            self.encoder.pretrain(data, validation_set=data)

        self.encoder.fit(data, data)

    def generate_encodings(self, data, labels, save_to_path):
        x = self.encoder.transform(data)
        x = x.astype(np.float32)
        np.save(save_to_path + '.npy', x)
        np.save(save_to_path + '_labels.npy', labels)
