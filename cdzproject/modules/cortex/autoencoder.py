class Autoencoder(object):

    def __init__(self):
        """A wrapper for an arbitrary autoencoder. Since we have pre-computed the encodings this object
        is just a shell."""
        pass

    def get_encoding(self, sensory_data):
        """Returns and encoding of the passed data"""
        return sensory_data

    def get_reconstruction(self, encoding):
        """Returns the reconstruction of the passed encoding. This is used for generative purposes"""
        return encoding
