import glob
import numpy as np
import argparse
import json
import yaml
from subset_finder import SubsetFinder
from submission_utils import read_image
from submission_utils import make_prediction
from submission_utils import read_val_image
import os
from constants import TTA

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

    all_month_models = args['all_models_paths']
    test_dir = args['test_dir']
    batch_size = args['batch_size']
    all_month_sizes = args['all_month_sizes']
    march_sizes = args['march_sizes']
    april_sizes = args['april_sizes']
    may_sizes = args['may_sizes']
    save_path = args['save_path']
    run_amount = args['run_amount']
    march_models = args['march_models']
    april_models = args['april_models']
    may_models = args['may_models']

    all_month_means = args['all_month_means']
    all_month_stds = args['all_month_stds']

    march_means = args['march_means']
    march_stds = args['march_stds']

    april_means = args['april_means']
    april_stds = args['april_stds']

    may_means = args['may_means']
    may_stds = args['may_stds']

    # subset chooser
    yaml_path = args['yaml_path']
    val_images = args['val_images']

    all_month_models *= TTA

    with open(yaml_path, 'r') as f:
        labels = yaml.safe_load(f)

    test_dir = np.array(glob.glob(f'{test_dir}/*.jpg'))
    all_images, march_images, april_images, may_images = read_image(test_dir)

    val_dir = []

    for path in val_images:
        val_dir += glob.glob(f'{path}/*.jpg')

    val_dir = np.array(val_dir)

    all_month_val_images, march_val_images, april_val_images, may_val_images = read_val_image(val_dir, labels)

    subset_finder_all_month = SubsetFinder(images_arr=all_month_val_images,
                                           yaml_path=yaml_path,
                                           model_paths=all_month_models,
                                           means=all_month_means,
                                           stds=all_month_stds,
                                           image_sizes=all_month_sizes,
                                           batch_size=batch_size,
                                           )

    subset_finder_march = SubsetFinder(images_arr=march_val_images,
                                       yaml_path=yaml_path,
                                       model_paths=march_models,
                                       means=march_means,
                                       stds=march_stds,
                                       image_sizes=march_sizes,
                                       batch_size=batch_size,
                                       )

    subset_finder_april = SubsetFinder(images_arr=april_val_images,
                                       yaml_path=yaml_path,
                                       model_paths=april_models,
                                       means=april_means,
                                       stds=april_stds,
                                       image_sizes=april_sizes,
                                       batch_size=batch_size,
                                       )
    subset_finder_may = SubsetFinder(images_arr=may_val_images,
                                     yaml_path=yaml_path,
                                     model_paths=may_models,
                                     means=may_means,
                                     stds=may_stds,
                                     image_sizes=may_sizes,
                                     batch_size=batch_size,
                                     )
    print(f'\n\n\n----- FINDING BEST ALL MONTH SUBSET -----\n\n\n')

    am_best_models, am_best_means, am_best_std = subset_finder_all_month()

    print(f'\n\n\n----- FINDING BEST MARCH SUBSET -----\n\n\n')

    march_best, march_best_mean, march_best_std = subset_finder_march()

    print(f'\n\n\n----- FINDING BEST APRIL SUBSET -----\n\n\n')

    april_best, april_best_mean, april_best_std = subset_finder_april()

    print(f'\n\n\n----- FINDING BEST MAY SUBSET -----\n\n\n')

    may_best, may_best_mean, may_best_std = subset_finder_may()

    predict_dict = {}

    print(f'\n\n---Running All Month Models---\n\n')

    predict_dict = make_prediction(predict_dict=predict_dict, models_paths=am_best_models, images=all_images,
                                   batch_size=batch_size, image_size=all_month_sizes, means=am_best_means,
                                   std=am_best_std, run_amount=run_amount)

    print(f'\n\n---Running March Models---\n\n')

    predict_dict = make_prediction(predict_dict=predict_dict, models_paths=march_best, images=march_images,
                                   batch_size=batch_size, image_size=march_sizes, means=march_best_mean,
                                   std=march_best_std, run_amount=run_amount)

    print(f'\n\n---Running April Modles---\n\n')

    predict_dict = make_prediction(predict_dict=predict_dict, models_paths=april_best, images=april_images,
                                   batch_size=batch_size, image_size=april_sizes, means=april_best_mean,
                                   std=april_best_std, run_amount=run_amount)
    print(f'\n\n---Running May Models---\n\n')

    predict_dict = make_prediction(predict_dict=predict_dict, models_paths=may_best, images=may_images,
                                   batch_size=batch_size, image_size=may_sizes, means=may_best_mean, std=may_best_std,
                                   run_amount=run_amount)

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
