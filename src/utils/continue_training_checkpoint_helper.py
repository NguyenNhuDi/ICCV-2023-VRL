import os
import json 
import torch.nn as nn
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

class ResumeTrainingHelper:
    def __init__(self, dataset_path, model_check):
        """
        Reads the trianing_config.json file for data like the model class, current epoch, and date.
        Also loads the weights.
        """
        self.dataset_path = dataset_path
        if os.path.exists(self.dataset_path + '/training_config.json'):
            with open(f'{self.dataset_path}/training_config.json', 'r') as file:
                ob = json.load(file)
                self.current_epoch = ob['current_epoch']
                self.model_class = ob['model_class']
                self.date = ob['date']
                self.best = ob['score']
                assert self.model_class == model_check, f'Your saved model is for class{self.model_class}, \
                        but you are trying to train with {model_check}.'
            self.weights = torch.load(f'{dataset_path}/model_progress/weights.pt')

        else:
            assert False, 'You have no saved models for this dataset! You can anot resume training.'
    
    @staticmethod
    def save_progress(params:dict, class_name:str, dataset_path:str, current_epoch:int=0, best_val_score:float=0):
        """
        Given a model and the current epoch, saves weights and relevant data to file.
        """
        import torch
        from datetime import datetime
        try:
            os.mkdir(f'{dataset_path}/model_progress')
        except:
            pass
        torch.save(params, f'{dataset_path}/model_progress/weights.pt')
        with open(f'{dataset_path}/model_progress/checkpoint.json', 'w') as fp:
            info = {
                'current_epoch':current_epoch,
                'model_class':str(class_name.__class__),
                'date' : str(datetime.now()),
                'score' : str(best_val_score)
            }
            son = json.dumps(info)
            fp.write(son)
        