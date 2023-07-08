import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision import models
import sys
import argparse
import json
from PIL import Image
import yaml

sys.path.append(r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\ICCV-2023-VRL')
from src.data_loading.DSAL import DSAL


def transform_image_label(image_path, label, transform):
    np_image = Image.open(image_path)

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
        np_image = transform(np_image)

    # converting the image and mask into tensors

    out_label = torch.tensor(out_label)

    return np_image, out_label


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

# TODO get val into its own folder
# TODO get val_batches
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
        transforms.Normalize((0, 0, 0),
                             (1, 1, 1))
    ])

    images = args['image_path']
    yaml_path = args['yaml_path']

    with open(yaml_path, 'r') as f:
        labels = yaml.safe_load(f)

    batch_size = args['batch_size']
    epochs = args['epochs']
    num_processes = args['num_processes']

    learning_rate = args['learning_rate']
    momentum = args['momentum']

    train_dsal = DSAL(images,
                labels,
                transform_image_label,
                batch_size=batch_size,
                epochs=epochs,
                num_processes=num_processes,
                max_queue_size=num_processes * 3,
                transform=transform)

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

    epoch_size = train_dsal.total_size // epochs

    counter = 0

    for i in range(train_dsal.num_batches):
        image, label = train_dsal.get_item()

        print(f'image shape: {image.shape}, label: {label.shape}')

        counter += 1

    train_dsal.join()
