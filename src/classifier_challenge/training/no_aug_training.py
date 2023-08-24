from model_trainer import ModelTrainer
import argparse
import json
import albumentations as A
import os
import pandas as pd
import yaml
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm


def lambda_transform(x: np.array, **kwargs) -> np.array:
    return x / 255


def __search__(x, l, r, arr):
    if l >= r:
        return -1

    m = (l + r) // 2

    if x == arr[m][1]:
        return m

    # item is to the left
    elif x < arr[m][1]:
        return __search__(x, l, m, arr)
    # item is to the right
    else:
        return __search__(x, m + 1, r, arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Model Trainer',
        description='This program will train a model',
        epilog='Vision Research Lab')
    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file.')
    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)

    yaml_path = args['yaml_path']
    best_save_name = args['best_save_name']
    last_save_name = args['last_save_name']
    save_dir = args['save_dir']

    try:
        os.makedirs(save_dir)
    except Exception:
        print(f'File already exists no need to create')

    image_size = args['image_size']
    batch_size = args['batch_size']
    epochs = args['epochs']
    num_processes = args['num_processes']
    learning_rate = args['learning_rate']
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    unfreeze_epoch = args['unfreeze_epoch']
    epoch_step = args['epoch_step']
    gamma = args['gamma']
    csv = args['csv']
    image_dir_20 = args['image_dir_20']
    image_dir_21 = args['image_dir_21']
    model_to_load = args['model_to_load']

    model = args['model']
    model_name = args['model_name']
    out_name = args['log_name']

    months = args['which_months']
    train = args['which_train_set']
    val = args['which_val_set']

    cut_mix = args['cut_mix']

    month_embedding_length = args['month_embedding_length']
    year_embedding_length = args['year_embedding_length']
    plant_embedding_length = args['plant_embedding_length']

    plant_index_path = args['plant_index']

    image_paths = glob.glob(f'{image_dir_20}/*.jpg')
    image_paths += glob.glob(f'{image_dir_21}/*.jpg')

    image_paths.sort(key=lambda x: os.path.basename(x))

    train_transform = A.Compose(
        transforms=[
            A.Resize(image_size, image_size),

            A.OneOf(
                transforms=[
                    A.Flip(p=0.5),
                    A.Rotate(
                        limit=(-90, 90),
                        interpolation=1,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                        always_apply=False,
                        p=0.75,
                    )
                ],
                p=0.3
            ),
            A.OneOf(
                transforms=[
                    A.Flip(p=0.5),
                    A.Rotate(
                        limit=(-90, 90),
                        interpolation=1,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                        always_apply=False,
                        p=0.75,
                    )
                ],
                p=0.3
            ),
            A.OneOf(
                transforms=[
                    A.Flip(p=0.5),
                    A.Rotate(
                        limit=(-90, 90),
                        interpolation=1,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                        always_apply=False,
                        p=0.75,
                    )
                ],
                p=0.3
            ),
            A.OneOf(
                transforms=[
                    A.Flip(p=0.5),
                    A.Rotate(
                        limit=(-90, 90),
                        interpolation=1,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                        always_apply=False,
                        p=0.75,
                    )
                ],
                p=0.3
            ),

            A.Lambda(image=lambda_transform)
        ],
        p=1.0,
    )

    val_transform = A.Compose(
        transforms=[
            A.Resize(image_size, image_size),
            A.Lambda(image=lambda_transform)
        ],
        p=1.0,
    )

    submit_json = {
        'test_dir': '',
        'save_path': '',
        'batch_size': '',
        'all_models_paths': [],
        'all_month_sizes': [],
        'all_month_means': [],
        'all_month_stds': [],
        'march_sizes': [],
        'april_sizes': [],
        'may_sizes': [],
        'march_means': [],
        'march_stds': [],
        'april_means': [],
        'april_stds': [],
        'may_means': [],
        'may_stds': [],
        'march_models': [],
        'april_models': [],
        'may_models': [],
        'run_amount': 1

    }

    with open(yaml_path, 'r') as f:
        yml_labels = yaml.safe_load(f)

    plant_indexes = None

    try:
        with open(plant_index_path, 'r') as f:
            plant_indexes = yaml.safe_load(f)
    except:
        print('plant index skipped')

    print(f'--- OPENING IMAGES ---')
    opened_images = []
    for path in tqdm(image_paths):
        image_name = os.path.basename(path)
        if image_name in yml_labels:
            temp = np.array(Image.open(path), dtype='uint8')
            opened_images.append((temp, image_name))

    for i in range(len(csv)):

        current_train_dict = {
            '0': {
                '3': {
                    'unfertilized': [],
                    '_PKCa': [],
                    'N_KCa': [],
                    'NP_Ca': [],
                    'NPK_': [],
                    'NPKCa': [],
                    'NPKCa+m+s': []
                },

                '4': {
                    'unfertilized': [],
                    '_PKCa': [],
                    'N_KCa': [],
                    'NP_Ca': [],
                    'NPK_': [],
                    'NPKCa': [],
                    'NPKCa+m+s': []
                },
                '5': {
                    'unfertilized': [],
                    '_PKCa': [],
                    'N_KCa': [],
                    'NP_Ca': [],
                    'NPK_': [],
                    'NPKCa': [],
                    'NPKCa+m+s': []
                }
            },
            '1': {
                '3': {
                    'unfertilized': [],
                    '_PKCa': [],
                    'N_KCa': [],
                    'NP_Ca': [],
                    'NPK_': [],
                    'NPKCa': [],
                    'NPKCa+m+s': []
                },

                '4': {
                    'unfertilized': [],
                    '_PKCa': [],
                    'N_KCa': [],
                    'NP_Ca': [],
                    'NPK_': [],
                    'NPKCa': [],
                    'NPKCa+m+s': []
                },
                '5': {
                    'unfertilized': [],
                    '_PKCa': [],
                    'N_KCa': [],
                    'NP_Ca': [],
                    'NPK_': [],
                    'NPKCa': [],
                    'NPKCa+m+s': []
                }
            }
        }

        df = pd.read_csv(csv[i])
        data_dict = df.to_dict(orient='list')

        for image_name in data_dict['train']:
            class_label = yml_labels[image_name]

            year = image_name[3]
            month = image_name[5]

            index = __search__(image_name, 0, len(opened_images), opened_images)

            current_train_dict[year][month][class_label].append(opened_images[index][0])

        trainer = ModelTrainer(labels=yml_labels,
                               current_train_dict=current_train_dict,
                               best_save_name=os.path.join(save_dir, best_save_name[i]),
                               last_save_name=os.path.join(save_dir, last_save_name[i]),
                               save_dir=save_dir,
                               csv=csv[i],
                               images=opened_images,
                               train_transform=train_transform,
                               val_transform=val_transform,
                               image_size=image_size,
                               submit_json=submit_json,
                               batch_size=batch_size,
                               epochs=epochs,
                               weight_decay=weight_decay[i],
                               num_processes=num_processes,
                               learning_rate=learning_rate[i],
                               momentum=momentum[i],
                               unfreeze_epoch=unfreeze_epoch,
                               epoch_step=epoch_step,
                               gamma=gamma[i],
                               model_to_load=model_to_load[i],
                               months=months,
                               val=val,
                               train=train,
                               model=model,
                               model_name=f'{model_name} -- {i + 1}',
                               out_name=out_name[i],
                               cutmix=cut_mix,
                               month_embedding_length=month_embedding_length,
                               year_embedding_length=year_embedding_length,
                               plant_embedding_length=plant_embedding_length,
                               plant_index=plant_indexes
                               )

        submit_json = trainer()

    with open(f'{save_dir}/{model_name}.json', 'w') as json_file:
        json.dump(submit_json, json_file, indent=4)