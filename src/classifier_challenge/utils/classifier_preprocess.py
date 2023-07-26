import math

import yaml
import argparse
import json
import numpy as np
import pandas as pd
import os


class LabelConverter:
    def __init__(self, img_dir,
                 test_paths,
                 save_pth,
                 save_name,
                 yml_path,
                 months,
                 vals,
                 trains,
                 split_percent=20):
        self.labels = {}
        for i in yml_path:
            with open(i, 'r') as file:
                self.labels.update(yaml.safe_load(file))

        self.image_dir = img_dir
        self.test_paths = test_paths
        self.save_path = save_pth
        self.save_name = save_name
        self.save_dict = {'train': [],
                          'val': []}
        self.split_percent = split_percent
        self.months = months
        self.vals = vals
        self.trains = trains

    def __call__(self):
        self.__label_to_csv__()

    def __label_to_csv__(self):
        labels_dict = {}

        # sort the images into labels to get even split
        for name in self.labels:
            if int(name[5]) not in self.months:
                continue
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
                curr_img = labels_dict[curr_class][index_counter]

                if int(curr_img[3]) in self.vals:
                    self.save_dict['val'].append(curr_img)

                counter += 1
                index_counter += 1

            # putting the rest of the images into train set
            counter = 0
            while counter < len(labels_dict[curr_class]) - val_len:
                curr_img = labels_dict[curr_class][index_counter]

                if int(curr_img[3]) in self.trains:
                    self.save_dict['train'].append(curr_img)
                counter += 1
                index_counter += 1
        # making the dictionary sub arrays lengths the same

        val_diff = len(self.save_dict['train']) - len(self.save_dict['val'])

        self.save_dict['val'] += [None for i in range(val_diff)]

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
    test_txt_path = args['test_texts']
    month_set = args['which_months']
    val_set = args['which_val_set']
    train_set = args['which_train_set']
    save_path = args['save_path']
    save_file_name = args['save_name']
    yaml_paths = args['yaml_paths']

    for name in save_file_name:
        converter = LabelConverter(image_dir,
                                   test_txt_path,
                                   save_path,
                                   name,
                                   yaml_paths,
                                   month_set,
                                   val_set,
                                   train_set)
        converter()
