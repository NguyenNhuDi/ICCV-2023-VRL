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


def generate(model_path, test_image_dir, file_name):
    img_dir = np.array(glob.glob(f'{test_image_dir}/*.jpg'))

    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize((0.4780, 0.4116, 0.3001),
                             (0.4995, 0.4921, 0.4583))
    ])

    model = torch.load(model_path, batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    image_batch = []
    name_batch = []

    counter = 0
    batch_counter = 0

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

    prediction = []

    for i in range(len(image_batch)):
        image = image_batch[i].to(device)
        output = model(image)

        for j in range(len(output)):
            name = name_batch[i][j]
            prediction = int(torch.argmax(output[j]).cpu().numpy())

        prediction.append((name, prediction))

    f = open(file_name, 'w')

    for i in prediction:
        f.write(f'{i[0]} {i[1]}\n')


if __name__ == '__main__':
    _20_save_name = 'predictions_WW2020.txt'
    _21_save_name = 'predictions_WR2021.txt'

    _20_model_path = r''
    _21_model_path = r''

    _20_test_dir = r'/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/test_image'
    _21_test_dir = r'/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/test_image'

    generate(_20_model_path, _20_test_dir, _20_save_name, 16)
    generate(_21_model_path, _21_test_dir, _21_save_name, 16)
