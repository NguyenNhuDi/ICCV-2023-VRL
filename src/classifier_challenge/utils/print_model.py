from torchvision import models

if __name__ == '__main__':
    model = models.efficientnet_v2_l(pretrained=True)

    print(model)

