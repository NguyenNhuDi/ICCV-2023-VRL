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

sys.path.append(r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\ICCV-2023-VRL')
from src.data_loading.DSAL import DSAL


# this is how you find mean and std
# image, label = train_dsal.get_item()
#
# channels_sum += torch.mean(image, dim=[0, 2, 3])
# channels_squared_sum += torch.mean(image ** 2, dim=[0, 2, 3])
# num_batches += 1
#
# counter += 1
#
# mean = channels_sum / num_batches
# std = (channels_sum / num_batches - mean ** 2) ** 0.5
#
# print(mean)
# print(std)


def transform_image_label(image_path, label, transform):
    out_image = Image.open(image_path)

    if label == 'unfertilized':
        out_label = 0
    elif label == '_PKCa':
        out_label = 1
    elif label == 'N_KCa':
        out_label = 2
    elif label == 'NP_Ca':
        out_label = 3
    elif label == 'NPK_':
        out_label = 4
    elif label == 'NPKCa':
        out_label = 5
    else:
        out_label = 6

    if transform is not None:
        out_image = transform(out_image)

    # converting the image and mask into tensors

    out_label = torch.tensor(out_label)

    return out_image, out_label


def freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def evaluate(model, val_batches, device, criterion, epoch):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0
    for batch in val_batches:
        image, label = batch
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(image)
            loss = criterion(outputs, label)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)
            _, prediction = outputs.max(1)

            total_correct += (label == prediction).sum()

    loss = total_loss / total
    accuracy = total_correct / total
    print(f'Epoch: {epoch}, Loss: {loss:6.4f}, Accuracy: {accuracy:6.4f}')


# TODO write train loop


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Echo Extractor',
        description='This program will train ICCV23 challenge with efficient net',
        epilog='Vision Research Lab')
    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file.')

    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)

    transform = transforms.Compose([
        transforms.RandomCrop((500, 500)),
        transforms.RandomRotation(180),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4780, 0.4116, 0.3001),
                             (0.4995, 0.4921, 0.4583))
    ])

    train_image = args['train_path']
    val_image = args['val_path']
    yaml_path = args['yaml_path']
    batch_size = args['batch_size']
    epochs = args['epochs']
    num_processes = args['num_processes']
    learning_rate = args['learning_rate']
    momentum = args['momentum']

    with open(yaml_path, 'r') as f:
        labels = yaml.safe_load(f)

    train_dsal = DSAL(train_image,
                      labels,
                      transform_image_label,
                      batch_size=batch_size,
                      epochs=epochs,
                      num_processes=num_processes,
                      max_queue_size=num_processes * 3,
                      transform=transform)

    val_dsal = DSAL(val_image,
                    labels,
                    transform_image_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    num_processes=num_processes,
                    max_queue_size=num_processes * 3,
                    transform=transform)

    val_dsal.start()

    # storing valid batches in memory

    val_batches = []
    for i in tqdm(range(val_dsal.num_batches)):
        val_batches.append(val_dsal.get_item())

    print('starting pathing...')
    train_dsal.start()
    print('pathing finished')

    # declaring the model
    model = models.efficientnet_b7(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=2560, out_features=256),
        nn.Linear(in_features=256, out_features=7)
    )

    freeze(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    counter = 0
    batches_per_epoch = train_dsal.num_batches // epochs
    epoch = 0

    total = 0
    total_correct = 0
    total_loss = 0
    for i in tqdm(range(train_dsal.num_batches)):

        if counter == batches_per_epoch:
            loss = total_loss / total
            accuracy = total_correct / total
            print(f'Epoch: {epoch}, Loss: {loss:6.4f}, Accuracy: {accuracy:6.4f}')

            total = 0
            total_correct = 0
            total_loss = 0
            epoch += 1
            counter = 0

        image, label = train_dsal.get_item()
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        outputs, _ = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total += image.size(0)
        _, predictions = outputs.max(1)
        total_correct += (predictions == label).sum()
        total_loss += loss.item() * image.size(0)

        counter += 1

    train_dsal.join()
