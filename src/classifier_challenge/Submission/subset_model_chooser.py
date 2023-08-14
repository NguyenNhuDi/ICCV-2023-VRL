from itertools import combinations
from submission_utils import make_single_prediction
import os
from tqdm import tqdm
import numpy as np


class SubsetModelChooser:

    def __init__(self, test_images: np.array,
                 labels: dict,
                 models,
                 mean,
                 std,
                 image_sizes,
                 batch_size=16,
                 run_amount=1,
                 score_constant=5):

        self.test_images = test_images
        self.labels = labels
        self.score_constant = score_constant

        self.models = []

        for i in range(len(models)):
            self.models += list(combinations(models, i + 1))

        self.mean = []

        for i in range(len(mean)):
            self.mean += list(combinations(mean, i + 1))

        self.std = []

        for i in range(len(std)):
            self.std += list(combinations(std, i + 1))

        self.image_sizes = []

        for i in range(len(image_sizes)):
            self.image_sizes += list(combinations(image_sizes, i + 1))
        self.batch_size = batch_size
        self.run_amount = run_amount
        self.length = len(self.models)

        self.dp = {}

    def __call__(self):

        if self.length <= 0:
            return [], [], []

        subset_performance = []

        for i in tqdm(range(self.length)):
            curr_model_subset = list(self.models[i])
            curr_mean_subset = list(self.mean[i])
            curr_std_subset = list(self.std[i])
            curr_size_subset = list(self.image_sizes[i])

            predict_dict = {}

            for j in range(len(curr_model_subset)):

                model_name = os.path.basename(curr_model_subset[j])

                if model_name in self.dp:

                    stored_dict = self.dp[model_name]

                    for key in stored_dict:

                        if key not in predict_dict:
                            predict_dict[key] = stored_dict[key]
                        else:
                            predict_dict[key] = [predict_dict[key][i] + stored_dict[key][i] for i in
                                                 range(len(predict_dict[key]))]


                else:

                    predict_dict = make_single_prediction(predict_dict=predict_dict,
                                                          image_size=curr_size_subset[j],
                                                          model_path=curr_model_subset[j],
                                                          mean=curr_mean_subset[j],
                                                          std=curr_std_subset[j],
                                                          images=self.test_images,
                                                          batch_size=self.batch_size,
                                                          run_amount=self.run_amount)
                    self.dp[model_name] = predict_dict

            score_achieved = self.__calculate_score__(predict_dict)
            score_total = len(predict_dict) * len(curr_model_subset) * self.score_constant

            grade = score_achieved / score_total

            subset_performance.append((grade, curr_model_subset, curr_mean_subset, curr_std_subset))

        best_score = -1e9
        best_index = -1

        index = 0
        for i in subset_performance:
            # print(i)/

            if i[0] > best_score:
                best_score = i[0]
                best_index = index

            index += 1

        print(
            f'score: {subset_performance[best_index][0]} --- num models: {len(subset_performance[best_index][1])}\n models:{subset_performance[best_index][1]}\n')

        best_models = subset_performance[best_index][1]
        best_means = subset_performance[best_index][2]
        best_stds = subset_performance[best_index][3]

        return best_models, best_means, best_stds

    def __calculate_score__(self, predict_dict):
        score = 0

        for name in predict_dict:
            correct_index = 0
            correct_class = self.labels[name]

            if correct_class == '_PKCa':
                correct_index = 1
            elif correct_class == 'N_KCa':
                correct_index = 2
            elif correct_class == 'NP_Ca':
                correct_index = 3
            elif correct_class == 'NPK_':
                correct_index = 4
            elif correct_class == 'NPKCa':
                correct_index = 5
            elif correct_class == 'NPKCa+m+s':
                correct_index = 6

            curr_item = predict_dict[name]

            for index in range(len(curr_item)):

                if index == correct_index:
                    score += self.score_constant * curr_item[index]
                else:
                    n = curr_item[index]
                    score -= (n * (n + 1)) / 2
        return score
