import random
import numpy as np
from PIL import Image
import torchvision.datasets as ds


class AnomalyMNIST(ds.MNIST):
    """Synthetically corrupts the MNIST images to introduce anomalies
    """

    def seed(self, index):
        random.seed(index)
        np.random.seed(index)

    def __getitem__(self, index: int):
        img = self.data[index].numpy()

        # Randomly introduce 'anomalies' by corrupting the input
        # we do this in a deterministic way so that the requested index number
        # it will always return the exact same permuted or clean image and label.
        self.seed(index)

        target = 0.0
        if random.random() < 0.5:
            sigma = random.random() * 50 + 50
            img = img.astype('float32') + np.random.normal(0, sigma, img.shape)
            img = np.clip(img, 0, 255).astype('uint8')
            target = 1.0

        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


