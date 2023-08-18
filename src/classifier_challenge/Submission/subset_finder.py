import json
import argparse
from submission_utils import read_val_image
from submission_utils import batch_images
from submission_utils import lambda_transform
import yaml
import glob
import numpy as np
import os
import torch
from tqdm import tqdm
import albumentations as A
from itertools import combinations
from scipy.stats import entropy
import functools


def sort_by_accuracy_then_entropy(a, b):
    if a[1][0] != b[1][0]:
        return a[1][0] > b[1][0]
    else:
        return a[1][1] > b[1][1]


class SubsetFinder:

    def __init__(self,
                 model_paths,
                 images_arr,
                 batch_size,
                 means,
                 stds,
                 image_sizes,
                 yaml_path):

        self.images_arr = images_arr
        self.batch_size = batch_size

        with open(yaml_path, 'r') as f:
            self.labels = yaml.safe_load(f)

        _, names = batch_images(images_arr=images_arr, transform=None, batch_size=1)

        self.img_names = [t[0] for t in names]

        self.combination_model_paths = []

        for i in range(1, len(model_paths)):
            self.combination_model_paths += list(combinations(model_paths, i))

        self.means = []

        for i in range(1, len(means)):
            self.means += list(combinations(means, i))

        self.stds = []

        for i in range(1, len(stds)):
            self.stds += list(combinations(stds, i))

        self.image_sizes = []

        for i in range(1, len(image_sizes)):
            self.image_sizes += list(combinations(image_sizes, i))

        self.dp = {}
        self.model_mean_stds_img_size = {}
        self.final = []

    def __call__(self):

        print(f'----- Begin Call -----')

        if len(self.combination_model_paths) == 0:
            return [], [], []

        for i in tqdm(range(len(self.combination_model_paths))):
            model_paths = self.combination_model_paths[i]

            prediction_list = [[] for i in range(len(self.images_arr))]

            for j in range(len(model_paths)):
                model_path = model_paths[j]
                mean = self.means[i][j]
                std = self.stds[i][j]
                img_size = self.image_sizes[i][j]
                self.model_mean_stds_img_size[model_path] = (mean, std, img_size)

                transform = A.Compose(
                    transforms=[
                        A.Resize(img_size, img_size),
                        A.Lambda(image=lambda_transform),
                        A.Normalize(mean=mean, std=std, max_pixel_value=1.0)
                    ],
                    p=1.0,
                )

                self.genereate_predictions(model_path=model_path,
                                           transform=transform,
                                           prediction_list=prediction_list)

            self.calculate_accuracy_and_entropy(prediction_list=prediction_list,
                                                curr_models=model_paths,
                                                curr_mean=self.means[i],
                                                curr_std=self.stds[i])

        # self.final = sorted(self.final, key=functools.cmp_to_key(sort_by_accuracy_then_entropy))
        self.final.sort(key=lambda x: x[1])

        self.final.reverse()

        high_score = self.final[0][1][0]
        out_index = 0
        while self.final[out_index][1][0] == high_score:
            out_index += 1

        out_index -= 1

        final_models = self.final[out_index][0]
        final_means = self.final[out_index][2]
        final_stds = self.final[out_index][3]

        print(f'accuracy: {self.final[out_index][1][0]} --- entropy: {self.final[out_index][1][1]}')
        print(f'Number of models: {len(final_models)}')

        return final_models, final_means, final_stds

    def genereate_predictions(self, model_path, transform, prediction_list, ):

        model_name = os.path.basename(model_path)

        if model_name in self.dp:
            models_prediction = self.dp[model_name]

            for i, prediction in enumerate(models_prediction):
                prediction_list[i].append(prediction)
        else:
            image_batch, _ = batch_images(images_arr=self.images_arr,
                                          transform=transform,
                                          batch_size=self.batch_size)

            curr_prediction = []

            model = torch.load(model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            index_counter = 0
            for i in range(len(image_batch)):
                image = image_batch[i].to(device)

                output = model(image)

                for j in range(len(output)):
                    prediction = int(torch.argmax(output[j]).cpu().numpy())

                    prediction_list[index_counter].append(prediction)
                    index_counter += 1

                    curr_prediction.append(prediction)
            self.dp[model_name] = curr_prediction

    @staticmethod
    def get_label(label):
        if label == 'unfertilized':
            return 0
        elif label == '_PKCa':
            return 1
        elif label == 'N_KCa':
            return 2
        elif label == 'NP_Ca':
            return 3
        elif label == 'NPK_':
            return 4
        elif label == 'NPKCa':
            return 5
        else:
            return 6

    def calculate_accuracy_and_entropy(self, prediction_list, curr_models, curr_mean, curr_std):
        total_predictions = len(curr_models)

        total_correct_score = 0
        total_wrong_score = 0
        total_images = len(self.img_names)

        for i, img_name in enumerate(self.img_names):
            correct_class = SubsetFinder.get_label(self.labels[img_name])

            correct_predictions = 0
            num_classes = [0 for x in range(7)]
            wrong_predictions = 0
            p_wrong_predictions = []

            for prediction in prediction_list[i]:
                if prediction == correct_class:
                    correct_predictions += 1
                else:
                    num_classes[prediction] += 1
                    wrong_predictions += 1

            for x in num_classes:
                if x != 0:
                    p_wrong_predictions.append(x / wrong_predictions)

            wrong_score = entropy(p_wrong_predictions)
            correct_score = correct_predictions / total_predictions

            total_correct_score += correct_score
            total_wrong_score += wrong_score

        average_correct = total_correct_score / total_images
        average_wrong = total_wrong_score / total_images
        scores = (average_correct, average_wrong)

        self.final.append((curr_models, scores, curr_mean, curr_std))


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

    val_dir = []

    with open(yaml_path, 'r') as f:
        labels = yaml.safe_load(f)

        for path in val_images:
            val_dir += glob.glob(f'{path}/*.jpg')

    val_dir = np.array(val_dir)

    all_month_val_images, march_val_images, april_val_images, may_val_images = read_val_image(val_dir, labels)

    march_finder = SubsetFinder(model_paths=march_models,
                                images_arr=march_val_images,
                                batch_size=batch_size,
                                means=march_means,
                                stds=march_stds,
                                image_sizes=march_sizes,
                                yaml_path=yaml_path,
                                csv_save_path=os.path.join(save_path, f'march_subsets.csv'))

    march_finder()

