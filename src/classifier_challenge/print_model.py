import torch
from torchvision import models

if __name__ == '__main__':
    model = models.squeezenet1_1(pretrained=True)

    print(model)
