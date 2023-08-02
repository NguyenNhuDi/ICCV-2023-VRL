from itertools import product

import torch
from torchvision import models
import albumentations as A
from PIL import Image
import numpy as np

import json as json

if __name__ == '__main__':
    # print('hello world')
    # model = models.resnext101_32x8d(pretrained=True)
    #
    # print(model)

    lr = [('lr', 0.1), ('lr', 0.01), ('lr', 0.05)]
    m = [('m', 0.8), ('m', 0.9), ('m', 0.7)]
    wd = [('wd', 0.001), ('wd', 0.000001), ('wd', 0.001), ('wd', 0.000001)]
    g = [('g', 0.85), ('g', 0.5)]

    out_json_0 = {"yaml_path": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/updated_yml.yml',
                  "best_save_name": [],
                  "last_save_name": [],
                  "save_dir": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/Grid_Searching',
                  "which_months": [3, 4, 5],
                  "which_val_set": [0, 1],
                  "which_train_set": [0, 1],
                  "image_size": 528,
                  "batch_size": 32,
                  "epochs": 53,
                  "weight_decay": [],
                  "num_processes": 20,
                  "learning_rate": [],
                  "momentum": [],
                  "unfreeze_epoch": 3,
                  "epoch_step": 10,
                  "gamma": [],
                  "csv": [],
                  "image_dir_20": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/images',
                  "image_dir_21": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/images',
                  "model_to_load": [],
                  "model": 'resnet152',
                  "model_name": 'EFFICIENT NET',
                  "log_name": []

                  }

    out_json_1 = {"yaml_path": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/updated_yml.yml',
                  "best_save_name": [],
                  "last_save_name": [],
                  "save_dir": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/Grid_Searching',
                  "which_months": [3, 4, 5],
                  "which_val_set": [0, 1],
                  "which_train_set": [0, 1],
                  "image_size": 528,
                  "batch_size": 32,
                  "epochs": 53,
                  "weight_decay": [],
                  "num_processes": 20,
                  "learning_rate": [],
                  "momentum": [],
                  "unfreeze_epoch": 3,
                  "epoch_step": 10,
                  "gamma": [],
                  "csv": [],
                  "image_dir_20": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/images',
                  "image_dir_21": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/images',
                  "model_to_load": [],
                  "model": 'resnet152',
                  "model_name": 'EFFICIENT NET',
                  "log_name": []

                  }

    out_json_2 = {"yaml_path": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/updated_yml.yml',
                  "best_save_name": [],
                  "last_save_name": [],
                  "save_dir": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/Grid_Searching',
                  "which_months": [3, 4, 5],
                  "which_val_set": [0, 1],
                  "which_train_set": [0, 1],
                  "image_size": 528,
                  "batch_size": 32,
                  "epochs": 53,
                  "weight_decay": [],
                  "num_processes": 20,
                  "learning_rate": [],
                  "momentum": [],
                  "unfreeze_epoch": 3,
                  "epoch_step": 10,
                  "gamma": [],
                  "csv": [],
                  "image_dir_20": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/images',
                  "image_dir_21": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/images',
                  "model_to_load": [],
                  "model": 'resnet152',
                  "model_name": 'EFFICIENT NET',
                  "log_name": []

                  }

    out_json_3 = {"yaml_path": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/updated_yml.yml',
                  "best_save_name": [],
                  "last_save_name": [],
                  "save_dir": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/Grid_Searching',
                  "which_months": [3, 4, 5],
                  "which_val_set": [0, 1],
                  "which_train_set": [0, 1],
                  "image_size": 528,
                  "batch_size": 32,
                  "epochs": 53,
                  "weight_decay": [],
                  "num_processes": 20,
                  "learning_rate": [],
                  "momentum": [],
                  "unfreeze_epoch": 3,
                  "epoch_step": 10,
                  "gamma": [],
                  "csv": [],
                  "image_dir_20": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/images',
                  "image_dir_21": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/images',
                  "model_to_load": [],
                  "model": 'resnet152',
                  "model_name": 'EFFICIENT NET',
                  "log_name": []

                  }

    all_combo = (list(product(lr, m, wd, g)))
    all_len = len(all_combo)

    counter = 0
    index = 0

    while counter < all_len//4:
        for i in all_combo[index]:
            if i[0] == 'lr':
                out_json_0['learning_rate'].append(i[1])
            elif i[0] == 'm':
                out_json_0['momentum'].append(i[1])
            elif i[0] == 'wd':
                out_json_0['weight_decay'].append(i[1])
            else:
                out_json_0['gamma'].append(i[1])

        out_json_0['best_save_name'].append('best')
        out_json_0['last_save_name'].append('last')
        out_json_0['csv'].append('/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/CSV/en_01.csv')
        out_json_0['log_name'].append(f'test{index}')
        out_json_0['model_to_load'].append('')

        index += 1
        counter += 1

    counter = 0
    while counter < all_len//4:
        for i in all_combo[index]:
            if i[0] == 'lr':
                out_json_1['learning_rate'].append(i[1])
            elif i[0] == 'm':
                out_json_1['momentum'].append(i[1])
            elif i[0] == 'wd':
                out_json_1['weight_decay'].append(i[1])
            else:
                out_json_1['gamma'].append(i[1])

        out_json_1['best_save_name'].append('best')
        out_json_1['last_save_name'].append('last')
        out_json_1['csv'].append('/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/CSV/en_01.csv')
        out_json_1['log_name'].append(f'test{index}')
        out_json_1['model_to_load'].append('')

        index += 1
        counter += 1

    counter = 0
    while counter < all_len//4:
        for i in all_combo[index]:
            if i[0] == 'lr':
                out_json_2['learning_rate'].append(i[1])
            elif i[0] == 'm':
                out_json_2['momentum'].append(i[1])
            elif i[0] == 'wd':
                out_json_2['weight_decay'].append(i[1])
            else:
                out_json_2['gamma'].append(i[1])

        out_json_2['best_save_name'].append('best')
        out_json_2['last_save_name'].append('last')
        out_json_2['csv'].append('/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/CSV/en_01.csv')
        out_json_2['log_name'].append(f'test{index}')
        out_json_2['model_to_load'].append('')

        index += 1
        counter += 1

    while index < len(all_combo):
        for i in all_combo[index]:
            if i[0] == 'lr':
                out_json_3['learning_rate'].append(i[1])
            elif i[0] == 'm':
                out_json_3['momentum'].append(i[1])
            elif i[0] == 'wd':
                out_json_3['weight_decay'].append(i[1])
            else:
                out_json_3['gamma'].append(i[1])

        out_json_3['best_save_name'].append('best')
        out_json_3['last_save_name'].append('last')
        out_json_3['csv'].append('/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/CSV/en_01.csv')
        out_json_3['log_name'].append(f'test{index}')
        out_json_3['model_to_load'].append('')

        index += 1

    json_string_0 = json.dumps(out_json_0, indent=2)

    with open('test_0.json', 'w') as out:
        out.write(json_string_0)

    json_string_1 = json.dumps(out_json_1, indent=2)

    with open('test_1.json', 'w') as out:
        out.write(json_string_1)

    json_string_2 = json.dumps(out_json_2, indent=2)

    with open('test_2.json', 'w') as out:
        out.write(json_string_2)

    json_string_3 = json.dumps(out_json_3, indent=2)

    with open('test_3.json', 'w') as out:
        out.write(json_string_3)
