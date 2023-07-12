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
import albumentations as A
import albumentations.pytorch
from DSAL import DSAL

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
    out_image = np.array(Image.open(image_path), dtype=np.float32) /255.0

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
        augmented = transform(image=out_image)
        out_image = augmented['image']
        


    # converting the image and mask into tensors

    out_image = torch.from_numpy(out_image).permute(2,0,1)
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
        label = label.type(torch.int64)
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(image)
            outputs = outputs.type(torch.float32)
            loss = criterion(outputs, label)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)
            _, prediction = outputs.max(1)

            total_correct += (label == prediction).sum()

    loss = total_loss / total
    accuracy = total_correct / total

    print(f'Evaluate --- Epoch: {epoch}, Loss: {loss:6.4f}, Accuracy: {accuracy:6.4f}')
    return loss, accuracy

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

    HEIGHT = 700
    WIDTH = 700
    transform = A.Compose(
        transforms=[
            A.RandomCrop(height=HEIGHT, width=WIDTH, always_apply=True),
            A.Resize(height=HEIGHT//2, width=WIDTH//2, always_apply=True),
            A.Flip(p=0.5),
            A.Rotate(
                limit=(-15, 15),
                interpolation=1,
                border_mode=0,
                value=0,
                mask_value=0,
                always_apply=False,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.5,
                saturation=0.5,
                hue=0.2,
                always_apply=False,
                p=0.5,
            ),
            A.ChannelShuffle(p=0.2),
            A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            # A.Normalize(mean=((0.4780, 0.4116, 0.3001)), std=(0.4995, 0.4921, 0.4583)),
        ],
        p=1.0,
    )
    train_image = args['train_path']
    val_image = args['val_path']
    yaml_path = args['yaml_path']
    batch_size = args['batch_size']
    epochs = args['epochs']
    num_processes = args['num_processes']
    learning_rate = args['learning_rate']
    momentum = args['momentum']
    unfreeze_epoch = args['unfreeze_epoch']
    epoch_step = args['epoch_step']
    gamma = args['gamma']
    best_save_name = args['best_save_name']
    last_save_name = args['last_save_name']

    with open(yaml_path, 'r') as f:
        labels = yaml.safe_load(f)

    val_dsal = DSAL(val_image,
                    labels,
                    transform_image_label,
                    batch_size=batch_size,
                    epochs=1,
                    num_processes=num_processes,
                    max_queue_size=num_processes * 2,
                    transform=transform)

    val_dsal.start()

    # storing valid batches in memory

    val_batches = []
    for i in tqdm(range(val_dsal.num_batches)):
        val_batches.append(val_dsal.get_item())

    val_dsal.join()

    train_dsal = DSAL(train_image,
                      labels,
                      transform_image_label,
                      batch_size=batch_size,
                      epochs=epochs,
                      num_processes=num_processes,
                      max_queue_size=num_processes * 2,
                      transform=transform)

    print('starting pathing...')
    train_dsal.start()
    print('pathing finished')

    # declaring the model
    model = models.efficientnet_b6(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=2304, out_features=256),
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

    best_loss = 1000
    best_accuracy = 0
    best_epoch = 0

    model.to(device)

    torch.set_grad_enabled(True)

    # scheduler: optimizer, step size, gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epoch_step, gamma)


    for i in tqdm(range(train_dsal.num_batches)):

        if counter == batches_per_epoch:
            total_loss = total_loss / total
            accuracy = total_correct / total
            print(f'Training --- Epoch: {epoch}, Loss: {total_loss:6.4f}, Accuracy: {accuracy:6.4f}')
            current_loss, current_accuracy = evaluate(model, val_batches, device, criterion, epoch)
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_epoch = epoch
                save_path = r'/home/adeeb.hossain1/Classification/saved_models/en_b6_1000_.pth'
                torch.save(model, save_path)

            if current_loss < best_loss:
                best_loss = current_loss
            
            print(f'Best epoch: {best_epoch}, Best Loss: {best_loss:6.4f}, Best Accuracy: {best_accuracy:6.4f}')
            # get_last_lr()
            model.train()

            total = 0
            total_correct = 0
            total_loss = 0
            epoch += 1
            counter = 0
            scheduler.step()

        if epoch == unfreeze_epoch:
            unfreeze(model)

        image, label = train_dsal.get_item()
        label = label.type(torch.int64)
        image, label = image.to(device), label.to(device)



        optimizer.zero_grad()
        outputs = model(image)

        # outputs = outputs.type(torch.float32)
        # outputs.dtype=float32
        loss = criterion(outputs, label)

        if epoch < unfreeze_epoch:
            loss.requires_grad = True

        loss.backward()
        optimizer.step()
        total += image.size(0)
        _, predictions = outputs.max(1)
        total_correct += (predictions == label).sum()
        total_loss += loss.item() * image.size(0)

        counter += 1

    total_loss = total_loss / total
    accuracy = total_correct / total

    print(f'Training --- Epoch: {epoch}, Loss: {total_loss:6.4f}, Accuracy: {accuracy:6.4f}')
    evaluate(model, val_batches, device, criterion, epoch)


    train_dsal.join()

    save_path = last_save_name
    torch.save(model, save_path)
