import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transform


class CustomTransformer:

    def __init__(self, image, mask, seed=123321):
        random.seed(seed)
        self.image = image
        self.mask = mask

    def random_horizontal_flip(self, chance=0.5):
        assert 0 <= chance <= 1, 'the chance has to be between 0 and 1'
        if random.random() <= chance:
            self.image = TF.hflip(self.image)
            self.mask = TF.hflip(self.mask)
        return self.image, self.mask

    def random_invert(self, chance=0.5):
        assert 0 <= chance <= 1, 'the chance has to be between 0 and 1'
        if random.random() <= chance:
            self.image = TF.invert(self.image)
            self.mask = TF.invert(self.mask)
        return self.image, self.mask

    def random_rotate(self, chance=0.5, angle=90):
        assert 0 <= chance <= 1, 'the chance has to be between 0 and 1'
        if random.random() <= chance:
            self.image = TF.rotate(self.image, angle)
            self.mask = TF.rotate(self.mask, angle)

        return self.image, self.mask

    def random_vertical_flip(self, chance=0.5):
        assert 0 <= chance <= 1, 'the chance has to be between 0 and 1'
        if random.random() <= chance:
            self.image = TF.vflip(self.image)
            self.mask = TF.vflip(self.mask)
        return self.image, self.mask

    def resize(self, size=(500, 500)):
        self.image = transform.Resize(self.image, size)
        self.mask = transform.Resize(self.mask, size)
        return self.image, self.mask
