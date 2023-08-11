import yaml
import argparse
import json
import numpy as np
import pandas as pd
import os


class LabelConverter:
    def __init__(self,
                 sv_dirs,
                 yml_path,
                 split_amount=5):
        labels = {}
        for i in yml_path:
            with open(i, 'r') as file:
                labels.update(yaml.safe_load(file))

        self.save_dirs = sv_dirs
        self.split_amount = split_amount

        self.march_items = {}
        self.april_items = {}
        self.may_items = {}

        self.num_architects = len(sv_dirs)

        for img_name in labels:
            curr_class = labels[img_name]

            if img_name[5] == '3':
                if curr_class not in self.march_items:
                    self.march_items[curr_class] = []
                self.march_items[curr_class].append(img_name)

            elif img_name[5] == '4':
                if curr_class not in self.april_items:
                    self.april_items[curr_class] = []
                self.april_items[curr_class].append(img_name)
            else:
                if curr_class not in self.may_items:
                    self.may_items[curr_class] = []
                self.may_items[curr_class].append(img_name)

        for classes in self.march_items:
            np.random.shuffle(self.march_items[classes])
            np.random.shuffle(self.april_items[classes])
            np.random.shuffle(self.may_items[classes])

    def __call__(self):
        self.__label_to_csv__()

    def __label_to_csv__(self):

        num_splits = len(self.save_dirs)

        all_train = [[] for i in range(num_splits * self.split_amount)]
        all_val = [[] for i in range(num_splits * self.split_amount)]

        """Splitting March and Storing it"""

        for c in self.may_items:

            curr_architect_chunks = []

            temp_chunks = np.array_split(self.march_items[c], num_splits)

            for i in range(num_splits):
                temp = np.array([])
                for j in range(num_splits):
                    if j == i:
                        continue

                    temp = np.concatenate((temp, temp_chunks[j]))

                curr_architect_chunks.append(list(temp))

            counter = 0
            for chunk in curr_architect_chunks:
                fold_chunks = np.array_split(chunk, self.split_amount)

                for i in range(self.split_amount):
                    val_set = fold_chunks[i]
                    train_set = np.array([])
                    for j in range(self.split_amount):
                        if j != i:
                            train_set = np.concatenate((fold_chunks[j], train_set))

                    all_train[counter] += list(train_set)
                    all_val[counter] += list(val_set)

                    counter += 1

        """Splitting April and Storing it"""

        for c in self.may_items:

            curr_architect_chunks = []

            temp_chunks = np.array_split(self.april_items[c], num_splits)

            for i in range(num_splits):
                temp = np.array([])
                for j in range(num_splits):
                    if j == i:
                        continue

                    temp = np.concatenate((temp, temp_chunks[j]))

                curr_architect_chunks.append(list(temp))

            counter = 0
            for chunk in curr_architect_chunks:
                fold_chunks = np.array_split(chunk, self.split_amount)

                for i in range(self.split_amount):
                    val_set = fold_chunks[i]
                    train_set = np.array([])
                    for j in range(self.split_amount):
                        if j != i:
                            train_set = np.concatenate((fold_chunks[j], train_set))

                    all_train[counter] += list(train_set)
                    all_val[counter] += list(val_set)

                    counter += 1

        """Splitting May and Storing it"""

        for c in self.may_items:

            curr_architect_chunks = []

            temp_chunks = np.array_split(self.may_items[c], num_splits)

            for i in range(num_splits):
                temp = np.array([])
                for j in range(num_splits):
                    if j == i:
                        continue

                    temp = np.concatenate((temp, temp_chunks[j]))

                curr_architect_chunks.append(list(temp))

            counter = 0
            for chunk in curr_architect_chunks:
                fold_chunks = np.array_split(chunk, self.split_amount)

                for i in range(self.split_amount):
                    val_set = fold_chunks[i]
                    train_set = np.array([])
                    for j in range(self.split_amount):
                        if j != i:
                            train_set = np.concatenate((fold_chunks[j], train_set))

                    all_train[counter] += list(train_set)
                    all_val[counter] += list(val_set)

                    counter += 1
        set_counter = 0

        for save_path in self.save_dirs:
            names = self.save_dirs[save_path]

            for name in names:

                curr_train = all_train[set_counter]
                curr_val = all_val[set_counter]

                while len(curr_val) != len(curr_train):
                    curr_val.append(None)

                csv_dict = {
                    'train': curr_train,
                    'val': curr_val
                }

                df = pd.DataFrame.from_dict(csv_dict)

                df.to_csv(os.path.join(save_path, name), index=False)

                set_counter += 1


if __name__ == "__main__":
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

    save_dirs = args['save_dirs']
    yaml_paths = args['yaml_paths']

    csv_converter = LabelConverter(sv_dirs=save_dirs, yml_path=yaml_paths)
    csv_converter()
