import argparse
import json
import os

import numpy as np


def read_prediction(file_path, predict_dict):
    with open(file_path, 'r') as predictions:
        for prediction in predictions:
            name, class_num = prediction.split()

            if name not in predict_dict:
                predict_dict[name] = [0 for i in range(7)]

            predict_dict[name][int(class_num)] += 1


def write_prediction(file, predict_dict):
    for key in predict_dict:
        prediction = np.argmax(predict_dict[key])
        file.write(f'{key} {prediction}\n')


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

    prediction_paths = args['predictions']
    save_dir = args['save_dir']

    _20_dict = {}
    _21_dict = {}

    for i in prediction_paths:
        # 2020
        curr_file_20 = os.path.join(i, 'predictions_WW2020.txt')
        curr_file_21 = os.path.join(i, 'predictions_WR2021.txt')

        read_prediction(curr_file_20, _20_dict)
        read_prediction(curr_file_21, _21_dict)

    f = open(os.path.join(save_dir, 'predictions_WW2020.txt'), 'w')

    write_prediction(f, _20_dict)

    f.close()

    f = open(os.path.join(save_dir, 'predictions_WR2021.txt'), 'w')

    write_prediction(f, _21_dict)

    f.close()
