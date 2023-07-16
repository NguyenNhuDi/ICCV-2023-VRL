import torch
from torchvision import models
import albumentations as A
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # model = models.squeezenet1_1(pretrained=True)
    #
    # print(model)

    transform = A.Compose(
        transforms=[
            A.Resize(128,128)
        ]
    )

    image = np.array(Image.open(r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\ICCV-2023-VRL\src\classifier_challenge\Screenshot 2023-07-16 010128.png'))

    augmented = transform(image=image)
    image = augmented['image']

    image = Image.fromarray(image)

    image.save('hehe.png')