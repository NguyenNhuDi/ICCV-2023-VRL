import numpy as np
import pandas as pd
import yaml
import json
import argparse
import os
from sklearn.model_selection import RepeatedKFold

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Winter Wheat, Winter Rye Classifier pre process',
        description='This program will create different val and train set for classification of winter wheat and '
                    'winter rye based on a  and hash function',
        epilog='Vision Research Lab')

    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)

    num_models = args['num_models']
    yaml_paths = args['yaml_paths']
    output_dir = args['output_dir']
    folds = args['folds']
    seed = args['seed']

    SIZE_PER_CLASS = 0.05

    if num_models == 1:
        SIZE_PER_CLASS = 0.2

    MARCH_MAX_SIZE = [192 * SIZE_PER_CLASS, 180 * SIZE_PER_CLASS, 192 * SIZE_PER_CLASS, 192 * SIZE_PER_CLASS,
                      192 * SIZE_PER_CLASS, 192 * SIZE_PER_CLASS, 192 * SIZE_PER_CLASS]
    APRIL_MAX_SIZE = [64 * SIZE_PER_CLASS, 60 * SIZE_PER_CLASS, 64 * SIZE_PER_CLASS, 64 * SIZE_PER_CLASS,
                      64 * SIZE_PER_CLASS, 64 * SIZE_PER_CLASS, 64 * SIZE_PER_CLASS]
    MAY_MAX_SIZE = [128 * SIZE_PER_CLASS, 120 * SIZE_PER_CLASS, 128 * SIZE_PER_CLASS, 128 * SIZE_PER_CLASS,
                    128 * SIZE_PER_CLASS, 128 * SIZE_PER_CLASS, 128 * SIZE_PER_CLASS]

    assert num_models > 0, f'{num_models} entered, at least 1 model must be used'

    unique_val_set = {}
    train_set = {}

    april_total = 0
    april_labels = [[] for i in range(7)]

    march_total = 0
    march_labels = [[] for i in range(7)]

    may_total = 0
    may_labels = [[] for i in range(7)]

    labels = {}

    for yaml_path in yaml_paths:
        with open(yaml_path, 'r') as f:
            labels.update(yaml.safe_load(f))

    temp = []

    for image_name in labels:
        temp.append((image_name, labels[image_name]))

    labels = temp

    for item in labels:
        image_name, curr_class = item

        class_index = 0

        if curr_class == '_PKCa':
            class_index = 1
        elif curr_class == 'N_KCa':
            class_index = 2
        elif curr_class == 'NP_Ca':
            class_index = 3
        elif curr_class == 'NPK_':
            class_index = 4
        elif curr_class == 'NPKCa':
            class_index = 5
        elif curr_class == 'NPKCa+m+s':
            class_index = 6

        month = int(image_name[5])

        if month == 3:
            if len(march_labels[class_index]) < MARCH_MAX_SIZE[class_index]:
                unique_val_set[image_name] = curr_class
            else:
                train_set[image_name] = curr_class

            march_labels[class_index].append(image_name)

        if month == 4:
            if len(april_labels[class_index]) < APRIL_MAX_SIZE[class_index]:
                unique_val_set[image_name] = curr_class
            else:
                train_set[image_name] = curr_class

            april_labels[class_index].append(image_name)

        if month == 5:
            if len(may_labels[class_index]) < MAY_MAX_SIZE[class_index]:
                unique_val_set[image_name] = curr_class
            else:
                train_set[image_name] = curr_class
            may_labels[class_index].append(image_name)

    if num_models < 2:

        out_csv = {
            'train': [],
            'val': []
        }

        for image_name in train_set:
            out_csv['train'].append(image_name)

        for image_name in unique_val_set:
            out_csv['val'].append(image_name)

        while len(out_csv['train']) != len(out_csv['val']):
            out_csv['val'].append(None)

        df = pd.DataFrame.from_dict(out_csv)
        df.to_csv(os.path.join(output_dir, 'single_model.csv'), index=False)

    else:
        train_list = []

        for image_name in train_set:
            train_list.append(image_name)

        rkf = RepeatedKFold(n_splits=folds, n_repeats=num_models, random_state=seed)

        model_counter = 0
        name_counter = 0

        for i, (train_index, val_index) in enumerate(rkf.split(X=np.zeros(len(train_set)))):
            out_csv = {
                'train': [],
                'val': []
            }

            for j in train_index:
                out_csv['train'].append(train_list[j])

            for j in val_index:
                out_csv['val'].append(train_list[j])

            while len(out_csv['train']) != len(out_csv['val']):
                out_csv['val'].append(None)

            df = pd.DataFrame.from_dict(out_csv)
            save_path = os.path.join(output_dir, f'model_{model_counter}')

            try:
                os.makedirs(save_path)
            except Exception:
                pass

            df.to_csv(os.path.join(save_path, f'{i}.csv'), index=False)

            name_counter += 1

            if name_counter == folds:
                model_counter += 1
                name_counter = 0

    # Make the yaml for submission

    with open(os.path.join(output_dir, 'unique_val.yml'), 'w') as f:
        yaml.dump(unique_val_set, f)