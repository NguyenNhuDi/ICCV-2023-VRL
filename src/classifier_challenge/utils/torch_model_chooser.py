from torchvision import models
from torch import nn
import warnings
import sys

warnings.filterwarnings("ignore")


class ModelChooser:
    def __init__(self, model_name):
        self.id = model_name

        # Efficient Net
        # TODO add the rest of efficient net family

        self.efficientnet_b0 = models.efficientnet_b0(pretrained=True)

        self.efficientnet_b0.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=256),
            nn.Linear(in_features=256, out_features=7)
        )

        self.efficientnet_b1 = models.efficientnet_b1(pretrained=True)

        self.efficientnet_b1.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=256),
            nn.Linear(in_features=256, out_features=7)
        )

        self.efficientnet_b6 = models.efficientnet_b6(pretrained=True)

        self.efficientnet_b6.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2304, out_features=256),
            nn.Linear(in_features=256, out_features=7)
        )

        # Google Net
        self.googlenet = models.googlenet(pretrained=True)

        self.googlenet.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.Linear(in_features=256, out_features=7)
        )

        # Resnet
        # TODO add the rest of resnet family

        self.resnet152 = models.resnet152(pretrained=True)

        self.resnet152.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),
            nn.Linear(in_features=256, out_features=7)
        )

        # ResNeXt
        # TODO add the rest of the resnext family

        self.resnext101_32x8d = models.resnext101_32x8d(pretrained=True)

        self.resnext101_32x8d.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),
            nn.Linear(in_features=256, out_features=7)
        )

        # Shuffle Net
        # TODO add the rest of the shuffle net family

        self.shufflenet_v2_x1_0 = models.shufflenet_v2_x1_0(pretrained=True)

        self.shufflenet_v2_x1_0.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.Linear(in_features=1024, out_features=7)
        )

        # TODO add the rest of the classification models on torch

    def __call__(self):
        return self.__choose_model__()

    def __choose_model__(self):

        # TODO add the rest of the ids, group them by family pls (and alphabetical)

        if self.id == 'efficientnet_b6':
            return self.efficientnet_b6
        elif self.id == 'googlenet':
            return self.googlenet
        elif self.id == 'resnet152':
            return self.resnet152
        elif self.id == 'resnext101_32x8d':
            return self.resnext101_32x8d
        elif self.id == 'shufflenet_v2_x1_0':
            return self.shufflenet_v2_x1_0
        else:
            sys.exit(f'Model: {self.id} is not part of the registered models')
