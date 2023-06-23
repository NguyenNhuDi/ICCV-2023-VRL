import os 
import json

class FingerprintManager:
    def __init__(self, dataset_path):
        """
        Just stores the fingerprint data in an easy access maner.
        """
        assert os.path.exists(f'{dataset_path}/fingerprint.json'), 'Generate fingerprint first.'
        with open(f'{dataset_path}/fingerprint.json', 'r') as file:
            ob = json.load(file)
            self.mean = ob['MEAN']
            self.std = ob['STD']
            self.num_classes = ob['unique_labels']