import pandas as pd
from utils.helpers import get_case_from_path
import numpy as np
import json

class FoldsManager:
 
    def __init__(self, dataset_path):
        """
        Used for the creation and loading of folds for training.
        Also an iterator - Nice.
        """
        self.dataset_path = dataset_path
        self.__loadfolds__()

    def __loadfolds__(self):
        """
        Loads the fold file and stores the data in splits.
        """
        with open(f'{self.dataset_path}/splits.json', 'r') as fp:
            self.splits = json.load(fp)
            print(f'Found {len(self.splits)} folds.')
            assert len(self.splits) > 0, 'There are no folds specified in your dataset.'
    
    def __iter__(self):
        """
        Does what it looks like...
        """
        self.current_split = 0
        return self

    def __next__(self):
        """
        Return the next split
        """
        if self.current_split < len(self.splits):
            self.current_split += 1
            return self.splits[self.current_split -1]
        else:
            raise StopIteration
    
    def getfold(self, fold:int):
        """
        Get a single fold.
        """
        return self.splits[str(fold)]
        
    @staticmethod
    def generate_k_folds(data:pd.DataFrame, dataset_path:str, k:int = 5):
        """
        Given a dataframe, generates a random k-fold file outlining k unique folds.
        """
        import json

        splits={}
        data = data['Image']

        frames = np.array_split(data, k)

        for i in range(k):
            splits[i] = {
                'Validation' : [],
                'Train'       : []
            }
            for p in frames[i]:
                splits[i]['Validation'].append(get_case_from_path(p))

            splits[i]['Train'] = sum([[get_case_from_path(p) for p in series] for idx, series in enumerate(frames) if idx != i], [])
        
        with open(f'{dataset_path}/splits.json', 'w') as fp:
            json.dump(splits, fp)
            print(f'Succesfully wrote split file to {fp.name}!')
    