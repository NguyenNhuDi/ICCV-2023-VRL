import argparse
import glob
import json
import os.path

import pandas as pd
import yaml
from PIL import Image
import numpy as np
import albumentations as A
from itertools import combinations
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Constants"""
MONTHS = [3, 4, 5]
CLASSES = ['unfertilized', '_PKCa', 'N_KCa', 'NP_Ca', 'NPK_', 'NPKCa', 'NPKCa+m+s']


def read_x_images(img_pth, x, csv_label, yml_label, class_name, month):
    num_images = len(img_pth)

    out_images = []

    counter = 0
    while len(out_images) < x and counter < num_images:
        img_name = os.path.basename(img_pth[counter])
        if img_name in csv_label:
            if yml_label[img_name] != class_name:
                counter += 1
                continue
        else:
            counter += 1
            continue
        if int(img_name[5]) != month:
            counter += 1
            continue

        out_images.append(np.array(Image.open(img_pth[counter])))

        counter += 1
    return out_images


def cut_mix(images_arr, tile_size=64, choose=2):
    num_images = len(images_arr)

    transforms = A.Compose(
        transforms=[
            A.Resize(1024, 1024)
        ],
        p=1.0,
    )

    # Resizing all the images

    for i in range(num_images):
        augmented = transforms(image=images_arr[i])
        images_arr[i] = augmented['image']

    all_combination = list(combinations(images_arr, choose))

    out_images = []

    for item in tqdm(all_combination):
        a, b = item
        i = 0
        while i < (1024 // tile_size) - 1:
            # print(f'i: {i}')
            j = 0
            while j < (1024 // tile_size) - 1:
                # print(f'j: {j}')
                first = a[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size, :].copy()
                second = b[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size, :].copy()
                b[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size, :] = first
                a[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size, :] = second

                j += 2
            i += 2

        out_images.append(a)
        out_images.append(b)
    return out_images


def save_images(class_type, img_to_save, yml_labels, csv_dict, _20_save_path, _21_save_path, year, month, counter):
    for img in img_to_save:
        img_name = f'202{year}0{month}_cut_mix_{counter}.jpg'
        save = Image.fromarray(img)
        yml_labels[img_name] = class_type
        csv_dict[len(csv_dict)] = img_name
        if year == 0:
            save.save(os.path.join(_20_save_path, img_name))
        else:
            save.save(os.path.join(_21_save_path, img_name))

        counter += 1
    return counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Winter Wheat, Winter Rye Classifier pre process',
        description='',
        epilog='Vision Research Lab')

    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)

    image_base_dir = args['image_base_dir']
    input_csv = args['input_csv']
    input_yml = args['input_yaml']
    tile_size = args['tile_size']
    _20_save_path = args['20_save_path']
    _21_save_path = args['21_save_path']
    image_per_class = args['image_per_class']
    choose = args['choose']

    image_paths = glob.glob(f'{image_base_dir[0]}/*.jpg')
    image_paths += glob.glob(f'{image_base_dir[1]}/*.jpg')

    csv_labels = pd.read_csv(input_csv).to_dict()

    for i in csv_labels:
        print(i)
        print(len(csv_labels[i]))

    with open(input_yml, 'r') as f:
        yml_labels = yaml.safe_load(f)

    counter = 0
    for _set in csv_labels:
        temp = csv_labels[_set]

        curr_set = []
        for i in temp:
            curr_set.append(temp[i])

        for class_type in CLASSES:
            for month in MONTHS:
                images_to_cut_mix = read_x_images(image_paths, image_per_class, curr_set, yml_labels, class_type, month)
                final_images = cut_mix(images_to_cut_mix, tile_size, choose)
                print(f'---{month}---')
                counter = save_images(class_type=class_type, img_to_save=final_images, csv_dict=temp,
                                      yml_labels=yml_labels,
                                      _20_save_path=_20_save_path, _21_save_path=_21_save_path, month=month,
                                      counter=counter, year=random.randint(0, 1))

    df = pd.DataFrame.from_dict(csv_labels)
    df.to_csv(input_csv, index=False)
