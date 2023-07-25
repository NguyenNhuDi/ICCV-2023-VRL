import math

import yaml
import argparse
import json
import numpy as np
import pandas as pd
import os


class LabelConverter:
    def __init__(self, img_dir,
                 test_path,
                 train_val_path,
                 save_pth,
                 save_name,
                 yml_path,
                 split_percent=20):

        with open(yml_path, 'r') as file:
            self.labels = yaml.safe_load(file)

        self.image_dir = img_dir
        self.test_path = test_path
        self.train_val_path = train_val_path
        self.save_path = save_pth
        self.save_name = save_name
        self.save_dict = {'train': [],
                          'val': [],
                          'test': []}
        self.split_percent = split_percent

    def __call__(self):
        self.__label_to_csv__()

    def __label_to_csv__(self):
        # putting the test images into the test set

        with open(self.test_path, 'r') as test_file:
            test_names = test_file

            for i in test_names:

                if '\n' in i:
                    i = i[:-1]
                self.save_dict['test'].append(i)

        labels_dict = {}

        # sort the images into labels to get even split

        for name in self.labels:
            curr_class = self.labels[name]
            if curr_class not in labels_dict:
                labels_dict[curr_class] = []
            labels_dict[curr_class].append(name)

        for curr_class in labels_dict:
            np.random.shuffle(labels_dict[curr_class])

            # putting the first 20% of images into val set

            counter = 0
            index_counter = 0

            val_len = math.floor((len(labels_dict[curr_class]) / 100) * self.split_percent)
            while counter < val_len:
                self.save_dict['val'].append(labels_dict[curr_class][index_counter])
                counter += 1
                index_counter += 1

            # putting the rest of the images into train set
            counter = 0
            while counter < len(labels_dict[curr_class]) - val_len:
                self.save_dict['train'].append(labels_dict[curr_class][index_counter])
                counter += 1
                index_counter += 1
        # making the dictionary sub arrays lengths the same

        val_diff = len(self.save_dict['train']) - len(self.save_dict['val'])
        test_diff = len(self.save_dict['train']) - len(self.save_dict['test'])

        self.save_dict['val'] += [None for i in range(val_diff)]
        self.save_dict['test'] += [None for i in range(test_diff)]

        df = pd.DataFrame(self.save_dict)
        df.to_csv(os.path.join(self.save_path, self.save_name), index=False)


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

    image_dir = args['img_dir']
    test_txt_path = args['test_text']
    train_val_txt_path = args['train_val_txt_path']
    save_path = args['save_path']
    save_file_name = args['save_name']
    yaml_path = args['yaml_path']

    for name in save_file_name:
        converter = LabelConverter(image_dir, test_txt_path, train_val_txt_path, save_path, name, yaml_path)
        converter()
