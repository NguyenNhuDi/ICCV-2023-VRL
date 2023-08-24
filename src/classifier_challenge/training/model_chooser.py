from torchvision import models
from torch import nn
import warnings
import sys
import torch


warnings.filterwarnings("ignore")


class ModelChooser:
    def __init__(self, model_name, month_embedding_length, year_embedding_length, plant_embedding_length,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.month_embedding_length = month_embedding_length
        self.year_embedding_length = year_embedding_length
        self.id = model_name
        self.plant_embedding_length = plant_embedding_length
        self.device = device

    def __call__(self):
        return self.__choose_model__()

    def __choose_model__(self):

        model = None
        # TODO add the rest of the ids, group them by family pls (and alphabetical)

        if self.id == 'alexnet':
            model = models.alexnet(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=9216, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'convnext_base':
            model = models.convnext_base(pretrained=True)

            model.classifier = nn.Sequential(
                # nn.LayerNorm((1024,), eps=1e-06, elementwise_affine=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=1024, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'densenet161':
            model = models.densenet161(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Linear(in_features=2208, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'densenet169':
            model = models.densenet169(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Linear(in_features=1664, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        # efficientnet
        elif self.id == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=1280, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )

        elif self.id == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=1280, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )

        elif self.id == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=1408, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=1536, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'efficientnet_b4':
            model = models.efficientnet_b4(pretraiend=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=1792, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'efficientnet_b5':
            model = models.efficientnet_b5(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=2048, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'efficientnet_b6':
            model = models.efficientnet_b6(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=2304, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )

        elif self.id == 'efficientnet_b7':
            model = models.efficientnet_b7(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=2560, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )
        elif self.id == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(pretrained=True)

            classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280 + self.month_embedding_length  + self.year_embedding_length
                          + self.plant_embedding_length,
                          out_features=1000),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1000, out_features=7)
            )

            model = SplittedModel(model=model, classifier=classifier,
                                  device=self.device,
                                  month_embedding_length=self.month_embedding_length,
                                  year_embedding_length=self.year_embedding_length,
                                  plant_embedding_length=self.plant_embedding_length)

        elif self.id == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m(pretrained=True)

            classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=1280 + self.month_embedding_length  + self.year_embedding_length
                          + self.plant_embedding_length
                          , out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

            model = SplittedModel(model=model, classifier=classifier,
                                  device=self.device,
                                  month_embedding_length=self.month_embedding_length,
                                  year_embedding_length=self.year_embedding_length,
                                  plant_embedding_length=self.plant_embedding_length)

        elif self.id == 'efficientnet_v2_l':
            model = models.efficientnet_v2_l(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=1280, out_features=1000),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1000, out_features=7)
            )
        # googlenet
        elif self.id == 'googlenet':
            model = models.googlenet(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=1024, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )

        elif self.id == 'inception_v3':
            model = models.inception_v3(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'mnasnet1_0':
            model = models.mnasnet1_0(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=1280, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Linear(in_features=576, out_features=1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1024, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        # resnet
        elif self.id == 'resnet18':
            model = models.resnet18(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=512, out_features=7)
            )

        elif self.id == 'resnet34':
            model = models.resnet34(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=512, out_features=7)
            )

        elif self.id == 'resnet50':
            model = models.resnet50(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'resnet101':
            model = models.resnet101(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'resnet152':
            model = models.resnet152(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )

        # resnext
        elif self.id == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=256),
                nn.Linear(in_features=256, out_features=7)
            )

        # shufflnet
        elif self.id == 'shufflenet_v2_x0_5':
            model = models.shufflenet_v2_x0_5(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'shufflenet_v2_x1_0':
            model = models.shufflenet_v2_x1_0(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'shufflenet_v2_x1_5':
            model = models.shufflenet_v2_x1_5(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'shufflenet_v2_x2_0':
            model = models.shufflenet_v2_x2_0(pretrained=True)

            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1000),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                # nn.AdaptiveAvgPool2d(output_size=(1, 1))
                nn.Flatten(),
                nn.Linear(in_features=57600, out_features=1000),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1000, out_features=512),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=512, out_features=7)
            )

        elif self.id == 'swin_s':
            model = models.swin_s()

            model.head = nn.Sequential(
                nn.Linear(in_features=768, out_features=7)
            )

        elif self.id == 'vgg16':
            model = models.vgg16(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=1000),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1000, out_features=7)
            )

        elif self.id == 'vit_b_32':
            model = models.vit_b_32(pretrained=True)

            model.classifier = nn.Sequential(
                nn.Linear(in_features=768, out_features=7)
            )
        elif self.id == 'three_month_baseline':

            """
            IMPLEMENT YOUR  CLASSIFIER LAYER HERE
            """
            classifier_layer = None
            """============================"""

            model = ThreeMonthBase(mar_path=r'base_line_models/env2_01_march2_best.pth',
                                   apr_path=r'base_line_models/env2_01_april0_best.pth',
                                   may_path=r'base_line_models/env2_01_may4_best.pth',
                                   classifier_layer=classifier_layer,
                                   device=self.device,
                                   month_embedding_length=self.month_embedding_length,
                                   year_embedding_length=self.year_embedding_length,
                                   plant_embedding_length=self.plant_embedding_length)

        else:
            sys.exit(f'Model: {self.id} is not part of the registered models')

        return model


class SplittedModel(nn.Module):
    def __init__(self, model, classifier, device, month_embedding_length, year_embedding_length, plant_embedding_length) -> None:
        super().__init__()

        self.device = device
        self.model = model
        self.classifier = classifier
        self.month_embedding_length = month_embedding_length
        self.year_embedding_length = year_embedding_length
        self.plant_embedding_length = plant_embedding_length

        names = [n for n, _ in model.named_children()]

        if names[-1] == 'fc':
            self.model.fc = nn.Identity()
        elif names[-1] == 'classifier':
            self.model.classifier = nn.Identity()

        self.model.to(device)
        self.classifier.to(device)

    def forward(self, img, month, year, plant):
        image_embedding = torch.flatten(self.model(img), start_dim=1)
        batch_size = image_embedding.shape[0]

        month_batch = []
        year_batch = []
        plant_batch = []

        for i in range(batch_size):
            month_batch.append((torch.randn(self.month_embedding_length) * 0.1) + month[i])
            year_batch.append((torch.randn(self.year_embedding_length) * 0.1) + year[i])
            plant_batch.append((torch.randn(self.plant_embedding_length) * 0.1) + (plant[i] / 100))

        month_batch = torch.stack(month_batch, dim=0).to(self.device)
        year_batch = torch.stack(year_batch, dim=0).to(self.device)
        plant_batch = torch.stack(plant_batch, dim=0).to(self.device)

        image_embedding = image_embedding.to(self.device)

        embedding = torch.cat((image_embedding, month_batch, year_batch, plant_batch), dim=1)

        return self.classifier(embedding)


class ThreeMonthBase(nn.Module):
    def __init__(self, mar_path, apr_path, may_path, classifier_layer, device, month_embedding_length,
                 year_embedding_length, plant_embedding_length):
        super().__init__()

        self.march_model = torch.load(mar_path)
        self.april_model = torch.load(apr_path)
        self.may_model = torch.load(may_path)
        self.classifier_layer = classifier_layer
        self.device = device
        self.month_embedding_length = month_embedding_length
        self.year_embedding_length = year_embedding_length
        self.plant_embedding_length = plant_embedding_length

        names = [n for n, _ in self.march_model.named_children()]

        if names[-1] == 'fc':
            self.march_model.fc = nn.Identity()
            self.april_model.fc = nn.Identity()
            self.may_model.fc = nn.Identity()
        elif names[-1] == 'classifier':
            self.march_model.classifier = nn.Identity()
            self.april_model.classifier = nn.Identity()
            self.may_model.classifier = nn.Identity()

        self.march_model.to(device)
        self.apr_model.to(device)
        self.may_model.to(device)
        self.classifier_layer.to(device)

    def forward(self, img, month, year, plant):
        image_embedding = torch.flatten(self.march_model(img), start_dim=1)

        if month == 4:
            image_embedding = torch.flatten(self.april_model(img), start_dim=1)
        elif month == 5:
            image_embedding = torch.flatten(self.may_model(img), start_dim=1)

        batch_size = image_embedding.shape[0]

        # month_batch = []
        # year_batch = []
        # plant_batch = []

        embedding_batch = []

        for i in range(batch_size):
            embedding_batch.append((torch.randn(self.month_embedding_length) * 0.1) + month[i])
            embedding_batch.append((torch.randn(self.year_embedding_length) * 0.1) + year[i])
            embedding_batch.append((torch.randn(self.plant_embedding_length) * 0.1) + (plant[i] / 100))

        embedding_batch = torch.stack(embedding_batch, dim=0).to(self.device)

        image_embedding = image_embedding.to(self.device)

        """
        ADD YOUR CODE HERE
        VVVVVVVVVVVVVVVVVVVVVVVVVV
        """

        embedding = None

        return self.classifier(embedding)


