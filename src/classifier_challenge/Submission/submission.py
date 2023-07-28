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
import albumentations as A


def read_image(image_dir):
    # out_image = np.array(Image.open(image_path), dtype='float32') / 255.0
    all_months = []
    march = []
    april = []
    may = []
    for i in tqdm(image_dir):
        image_name = os.path.basename(i)
        curr_item = (np.array(Image.open(i), dtype='uint8'), image_name)
        all_months.append(curr_item)

        if image_name[5] == '3':
            march.append(curr_item)
        elif image_name[5] == '4':
            april.append(curr_item)
        else:
            may.append(curr_item)

    return all_months, march, april, may


def process_image(image, transform):
    out_image = image[0]

    image_name = image[1]

    if transform is not None:
        augmented = transform(image=out_image)
        out_image = augmented['image']

    # converting the image and mask into tensors

    out_image = torch.from_numpy(out_image).permute(2, 0, 1)
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


def generate(model_path, images_arr, batch_size, transform, predict_dict):
    model = torch.load(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    image_batch = []
    name_batch = []

    counter = 0
    batch_counter = 0

    temp_img = []
    temp_name = []

    for i in tqdm(range(len(images_arr))):
        image, image_name = process_image(images_arr[i], transform)

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

    if len(temp_img) > 0:
        temp_img = torch.stack(temp_img, dim=0)

        image_batch.append(temp_img)
        name_batch.append(temp_name)

    for i in tqdm(range(len(image_batch))):
        image = image_batch[i].to(device)
        output = model(image)

        for j in range(len(output)):
            name = name_batch[i][j]
            prediction = int(torch.argmax(output[j]).cpu().numpy())

            if name in predict_dict:
                predict_dict[name][prediction] += 1
            else:
                predict_dict[name] = [0 for i in range(7)]
                predict_dict[name][prediction] += 1

    return predict_dict


def make_prediction(predict_dict, models_paths, images, batch_size, transform):
    counter = 0
    for model_path in models_paths:

        print(f'\n\n\n ----curr model {os.path.basename(model_path)}---\n\n')
        counter += 1

        for i in range(run_amount):
            print(f'\n\n ---iteration {i}---\n\n')

            predict_dict = generate(model_path, images, batch_size, transform, predict_dict)

    return predict_dict


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

    all_model_paths = args['all_models_paths']
    test_dir = args['test_dir']
    batch_size = args['batch_size']
    height = args['height']
    width = args['height']
    save_path = args['save_path']
    run_amount = args['run_amount']
    march_models = args['march_models']
    april_models = args['april_models']
    may_models = args['may_models']

    test_dir = np.array(glob.glob(f'{test_dir}/*.jpg'))
    all_images, march_images, april_images, may_images = read_image(test_dir)

    predict_dict = {}

    transform = A.Compose(
        transforms=[
            A.RandomCrop(750, 750),
            A.Resize(height, width),
            # A.Normalize(mean=((0.5385, 0.4641, 0.3378)), std=(0.5385, 0.4641, 0.3378))
            A.Normalize(mean=((0.4680, 0.4038, 0.2885)), std=(0.2476, 0.2107, 0.1931))

        ],
        p=1.0,
    )

    print(f'\n\n---Running All Month Models---\n\n')

    predict_dict = make_prediction(predict_dict, all_model_paths, all_images, batch_size, transform)

    print(f'\n\n---Running March Models---\n\n')

    predict_dict = make_prediction(predict_dict, march_models, march_images, batch_size, transform)

    print(f'\n\n---Running April Modles---\n\n')

    predict_dict = make_prediction(predict_dict, april_models, april_images, batch_size, transform)

    for name in predict_dict:
        print(f'{name} --- {predict_dict[name]}')

    print(f'\n\n---Running May Models---\n\n')

    predict_dict = make_prediction(predict_dict, may_models, may_images, batch_size, transform)

    predictions_20 = []
    predictions_21 = []

    for key in predict_dict:
        curr_item = np.array(predict_dict[key])

        prediction = curr_item.argmax()

        if key[0:4] == '2020':
            predictions_20.append((key, prediction))
        else:
            predictions_21.append((key, prediction))

    f = open(os.path.join(save_path, 'predictions_WW2020.txt'), 'w')

    for i in predictions_20:
        f.write(f'{i[0]} {i[1]}\n')

    f.close()

    f = open(os.path.join(save_path, 'predictions_WR2021.txt'), 'w')

    for i in predictions_21:
        f.write(f'{i[0]} {i[1]}\n')

    f.close()
