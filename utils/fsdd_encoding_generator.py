from free_spoken_digit_dataset.utils.fsdd import FSDD
from cdzproject.modules.autoencoder.autoencoder import Autoencoder


def generate_encodings():
    """Generates encodings for the fsdd dataset."""
    images, labels = FSDD.get_spectrograms()
    labels = [int(label) for label in labels]

    # Split into train and test set.
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for idx in range(10):
        # Extract train dataset
        start = idx * 50 + 5
        end = start + 45
        train_images.extend(images[start:end])
        train_labels.extend(labels[start:end])

        # Extract test dataset
        start = idx * 50
        end = start + 5
        test_images.extend(images[start:end])
        test_labels.extend(labels[start:end])

    # Initialize the autoencoder
    # We need many epochs because the the training set is so small and the learning rate low.
    autoencoder = Autoencoder([4096, 256, 64], pretrain=True, pretrain_epochs=20, finetune_epochs=500, finetune_batch_size=16)
    autoencoder.train(train_images)

    autoencoder.generate_encodings(train_images, train_labels, save_to_path='./data/fsdd_train_encodings_2')
    autoencoder.generate_encodings(test_images, test_labels, save_to_path='./data/fsdd_test_encodings_2')


if __name__ == '__main__':
    generate_encodings()
