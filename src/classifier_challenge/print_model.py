import torch
from torchvision import models
import albumentations as A
from PIL import Image
import numpy as np

if __name__ == '__main__':
    model = models.shufflenet_v2_x1_0(pretrained=True)

    print(model)

