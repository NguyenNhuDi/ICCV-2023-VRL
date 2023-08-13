import glob
import numpy as np
import argparse
import json
import yaml
from subset_model_chooser import SubsetModelChooser
from submission_utils import read_image


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
    unique_val_images = args['unique_val_images']

    with open(yaml_path, 'r') as f:
        labels = yaml.safe_load(f)

    test_dir = np.array(glob.glob(f'{test_dir}/*.jpg'))
    all_images, march_images, april_images, may_images = read_image(test_dir)

    val_dir = np.array(glob.glob(f'{unique_val_images}/*.jpg'))
    all_val_images, march_val_images, april_val_images, may_val_images = read_image(val_dir)

    subset_finder = SubsetModelChooser(test_images=march_val_images,
                                       labels=labels,
                                       models=march_models,
                                       mean=march_means,
                                       std=march_stds,
                                       image_sizes=march_sizes,
                                       subset_size=4,
                                       batch_size=batch_size
                                       )
    subset_finder()

    # predict_dict = {}
    #
    # print(f'\n\n---Running All Month Models---\n\n')
    #
    # predict_dict = make_prediction(predict_dict=predict_dict, models_paths=all_model_paths, images=all_images,
    #                                batch_size=batch_size, image_size=all_month_sizes, means=all_month_means,
    #                                std=all_month_stds)
    #
    # print(f'\n\n---Running March Models---\n\n')
    #
    # predict_dict = make_prediction(predict_dict=predict_dict, models_paths=march_models, images=march_images,
    #                                batch_size=batch_size, image_size=march_sizes, means=march_means, std=march_stds)
    #
    # print(f'\n\n---Running April Modles---\n\n')
    #
    # predict_dict = make_prediction(predict_dict=predict_dict, models_paths=april_models, images=april_images,
    #                                batch_size=batch_size, image_size=april_sizes, means=april_means, std=april_stds)
    # print(f'\n\n---Running May Models---\n\n')
    #
    # predict_dict = make_prediction(predict_dict=predict_dict, models_paths=may_models, images=may_images,
    #                                batch_size=batch_size, image_size=may_sizes, means=may_means, std=may_stds)
    #
    # predictions_20 = []
    # predictions_21 = []
    #
    # for key in predict_dict:
    #     curr_item = np.array(predict_dict[key])
    #
    #     prediction = curr_item.argmax()
    #
    #     if key[0:4] == '2020':
    #         predictions_20.append((key, prediction))
    #     else:
    #         predictions_21.append((key, prediction))
    #
    # f = open(os.path.join(save_path, 'predictions_WW2020.txt'), 'w')
    #
    # for i in predictions_20:
    #     f.write(f'{i[0]} {i[1]}\n')
    #
    # f.close()
    #
    # f = open(os.path.join(save_path, 'predictions_WR2021.txt'), 'w')
    #
    # for i in predictions_21:
    #     f.write(f'{i[0]} {i[1]}\n')
    #
    # f.close()
