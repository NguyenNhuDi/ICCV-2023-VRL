import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import SimpleITK as sitk

class FingerprintExtractor:
    
    def __init__(self, dataset_path, dataframe):
        """
        This class is for building the fingerprint of a dataset.
        """
        self.dataset_path = dataset_path
        self.dataframe = dataframe
    
    def write(self):
        """
        Write the fingerprint to dataset folder.
        """
        mean, std, num_labels = self._getmeanstd_num_labels()

        print = {
            'MEAN': mean,
            'STD' : std,
            'unique_labels':num_labels
        }

        with open(f'{self.dataset_path}/fingerprint.json', 'w') as file:
            json.dump(print, file)
            
    def _getmeanstd_num_labels(self):
        """
        Reads the mean and std of a dataset.
        """
        mean = 0
        std = 0
        unique_labels = np.array([])

        for _, row in tqdm(self.dataframe.iterrows(), total=self.dataframe.shape[0]):
            im = row['Object Address'].get_image_array(image_only=True)
            mask = row['Object Address'].get_segmentation_array(segmentation_only=True)
            
            unique_labels = np.unique(np.append(np.unique(mask), unique_labels))

            mean += np.mean(im)
            std += np.std(im)

        mean /= len(self.dataframe)
        std /= len(self.dataframe)
        return mean, std, len(unique_labels)