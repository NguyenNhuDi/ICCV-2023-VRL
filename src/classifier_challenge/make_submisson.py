import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision import models
import sys
import argparse
import json
from PIL import Image
import yaml
from tqdm import tqdm
from DSAL import DSAL
import os


def read_image(image_path, transform):
    out_image = Image.open(image_path)
    image_name = os.path.basename(image_path)

    if transform is not None:
        out_image = transform(out_image)

    return out_image, image_name


def evaluate(model, val_batches, device):
    model.eval()
    # total_correct = 0
    # total_loss = 0
    # total = 0
    for batch in val_batches:
        image, label = batch
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(image)
            for i in range(len(outputs)):
                prediction_index = torch.argmax(outputs[i]).cpu().numpy()

    # loss = total_loss / total
    # accuracy = total_correct / total


if __name__ == '__main__':
    load_path = r'/home/nhu.nguyen2/ICCV_2023/classifier_challenge/en_b60.pth'
    test_image_dir = r'/home/nhu.nguyen2/ICCV_2023/classifier_challenge/test_image'

    img_dir = np.array(glob.glob(f'{test_image_dir}/*.jpg'))

    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize((0.4780, 0.4116, 0.3001),
                             (0.4995, 0.4921, 0.4583))
    ])

    model = torch.load(load_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    image_batch = []
    name_batch = []

    counter = 0
    batch_counter = 0
    batch_size = 32

    temp_img = []
    temp_name = []

    for i in tqdm(range(len(img_dir))):
        image, image_name = read_image(img_dir[i], transform=transform)

        temp_img.append(image)
        temp_name.append(image_name)

        batch_counter += 1

        if batch_counter == batch_size:
            temp_img = torch.stack(temp_img, dim=0)

            image_batch.append(temp_img)
            name_batch.append(temp_name)
            temp_img = []
            temp_name = []

            batch_counter = 0

    temp_img = torch.stack(temp_img, dim=0)

    image_batch.append(temp_img)
    name_batch.append(temp_name)

    for i in range(len(image_batch)):
        image = image_batch[i].to(device)
        output = model(image)

        for j in range(len(output)):
            print(name_batch[i][j], torch.argmax(output[j]).cpu().numpy())
