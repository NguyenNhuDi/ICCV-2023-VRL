from itertools import combinations
from submission_utils import make_prediction

import numpy as np


class SubsetModelChooser:

    def __init__(self, test_images: np.array,
                 labels: dict,
                 models,
                 mean,
                 std,
                 image_sizes,
                 subset_size=5,
                 batch_size=16,
                 run_amount=1):
        self.test_images = test_images
        self.labels = labels
        self.models = list(combinations(models, subset_size))
        self.mean = list(combinations(mean, subset_size))
        self.std = list(combinations(std, subset_size))
        self.image_sizes = list(combinations(image_sizes, subset_size))
        self.batch_size = batch_size
        self.run_amount = run_amount
        self.length = len(self.models)

    def __call__(self):
        for i in range(self.length):
            curr_model_subset = list(self.models[i])
            curr_mean_subset = list(self.mean[i])
            curr_std_subset = list(self.std[i])
            curr_size_subset = list(self.image_sizes[i])

            predict_dict = {}

            predict_dict = make_prediction(predict_dict=predict_dict,
                                           image_size=curr_size_subset,
                                           models_paths=curr_model_subset,
                                           means=curr_mean_subset,
                                           std=curr_std_subset,
                                           images=self.test_images,
                                           batch_size=self.batch_size,
                                           run_amount=self.run_amount)

            print(predict_dict)
