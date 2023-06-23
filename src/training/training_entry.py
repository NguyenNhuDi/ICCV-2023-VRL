import sys
sys.path.append('/home/student/andrew/Documents/Seg3D')
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import os, time
from src.utils.continue_training_checkpoint_helper import ResumeTrainingHelper
from src.utils.find_class_by_name import my_import
from src.data_loading.data_helper import get_dataframe_fram_dataset, get_dataset_from_dataframe
from src.training.augmentations import get_train_augmentations, get_validation_transformations
from src.utils.folds_manager import FoldsManager
from src.utils.fingerprint_helper import FingerprintManager
from torch.optim.lr_scheduler import ExponentialLR, PolynomialLR
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
from tqdm import tqdm
from models.model_generator import ModelGenerator
import pandas as pd
import argparse
from monai.data import DataLoader, set_track_meta, decollate_batch
from monai.transforms import AsDiscrete, Compose, EnsureType
import threading
from monai.metrics import DiceMetric
from torch.cuda.amp import GradScaler as scaler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from monai.inferers import sliding_window_inference
import torchio as tio
import logging 
from loss.diceceloss import DiceLoss
import numpy as np
from training.loss.scheduler import PolyLRScheduler

def get_model_from_str(model_description:str)->nn.Module:
    """
    Either loads model class from name, or dynamically generates json.
    """
    if not 'json' in model_description:
        if model_description == 'nnUNet':
            from src.models.generate_nnunet_model import get_nnunet
            model = nn.Sequential(
                get_nnunet(),
                nn.Sigmoid()
            )
            return model
        return my_import(model_description)()
    
    return ModelGenerator(model_description).get_model()


class Trainer:

    def __init__(self, 
                 dataset_path, 
                 fold,
                 device:int,
                 treat_2d:bool=False,
                 lr:float = 0.001,
                 batch_size:int=2,
                 loss_function:str='dice',
                 world_size:int = 2,
                 mixed_precision:bool=True,
                 log:str = None,
                 use_sgd = False
            ) -> None:
        """
        This class is to manage the training of a dataset.
        The model class will be searched for in ./models, 
        unless it is a json file, in whoch case it will be generated dynamically.
        You can resume training if there exists a training_config.json file in the dataset folder.
        You can pick which fold to train, or if to train all folds sequentially. Will be adding concurrency soon.
        """
        
        self.dataset_path = dataset_path
        self.world_size = world_size

        self.__define_data__(fold)

        self.mixed_precision = mixed_precision
        self.treat_2d = treat_2d
        self.lr = lr
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.rank = device
        self.device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
        self.patch_size = [100, 256, 256]
        fingerprint_helper = FingerprintManager(dataset_path=dataset_path)
        self.fingerprint_helper = fingerprint_helper     
        self.num_labels = fingerprint_helper.num_classes 
        self.post_pred = AsDiscrete(threshold=0.5)     #For inference stuff
        self.best_score = 0
        self.use_sgd = use_sgd
        self.isnnunet = False

        self.initialize_logging(log)
        self._print(f"Working on {self.device}")

    def __define_data__(self, fold):
        dataframe = get_dataframe_fram_dataset(self.dataset_path)
        validation = fold['Validation']
        train = fold['Train']
            
        self.validation_dataframe = dataframe.loc[dataframe['Case'].isin(validation)]
        self.train_dataframe = dataframe.loc[dataframe['Case'].isin(train)]

    def _getdataloader(self, is_validation:bool=False)->DataLoader:
        """
        Given a dataframe, returns a dataloader.
        """
        if not is_validation:
            ds = get_dataset(self.patch_size, is_validation, self.train_dataframe, self.dataset_path)
        else:
            ds = get_dataset(self.patch_size, is_validation, self.validation_dataframe, self.dataset_path)
        
        return DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            sampler = DistributedSampler(
                dataset=ds,
                shuffle=not is_validation,
            )
        )

    def _resume_training_entry(self, epochs:int) -> None:
        """
        Loads model used last time for training, as well as current epoch and the weights.
        """
        helper = ResumeTrainingHelper(self.dataset_path, self.model.__class__)

        self.model = get_model_from_str(helper.model_class)
        if(helper.model_class == 'nnUNet'):
            self.isnnunet = True
        self.current_epoch = helper.current_epoch
        self.weights = helper.weights
        self.best_score = helper.best
        assert helper.current_epoch < epochs, \
            f'You stated that you want to load progress, which was last on' + \
                    f'epoch #{helper.current_epoch}. You then stated that you want to run {epochs} epochs.' + \
                    'You\'ve already done that. Aborting.'
        
        self._print(f'Continuing with {self.model.__class__}, epoch#{self.current_epoch}, last trained on {helper.date}.')
        

    def train_entry(self, 
                    model:str = None, 
                    epochs:int = 100, 
                    resume_training:bool=False, 
                    save_best:bool=True, 
                ) -> None:
        """
        Loads important things like the model class, and if applicable reads the training progress.
        Starts the training loop.
        """
        self.save_best = save_best #Wether or not we care about this run.
        if resume_training:
            self._resume_training_entry(epochs) #Load the goods. Will crash if relevant files dont exist.
        else:        
            self.model = get_model_from_str(model)
            self.current_epoch = 0
        
        if 'cuda' in self.device:
            self.model = self.model.to(device=self.device)
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=self.isnnunet)
        else:
            self.model = DDP(self.model) 
        if resume_training:
            self.model.load_state_dict(self.weights)
        
        if self.rank == 0:
            self._print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        #If folds is -1, run all folds.
        self._train(epochs, self._getdataloader(is_validation=False), self._getdataloader(is_validation=True))   

    def _loss_and_calculate_out(self, inputs:torch.Tensor, target:torch.Tensor, loss_function)->torch.Tensor:
        outputs = self.model(inputs)
        return loss_function(outputs, target)


    def _runvalidation(self, batch:torch.Tensor, loss_function):
         
        inputs, labels = self.__preparebatch__(batch)
        
        outputs = self.model(inputs)
        
        return DiceLoss.dice_score(self.post_pred(outputs), labels), loss_function(outputs, labels)

    def _trainstep(self, batch:torch.Tensor, loss_function, optimizer:torch.optim = None)->float:
        inputs, labels = self.__preparebatch__(batch)
        optimizer.zero_grad()

        if self.mixed_precision and torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = self._loss_and_calculate_out(inputs, labels, loss_function)
        else:
            loss = self._loss_and_calculate_out(inputs, labels, loss_function)

        loss.backward()
        optimizer.step()

        return loss

    def _train(self, epochs:int, train_dataloader, validation_dataloader):
        """
        Trains model using Adam optimizer, and dice loss.
        """
        
        current_epoch = self.current_epoch
        set_track_meta(False)
        
        self._print(f"Working on {self.device}.") 

        #============================================Training loop starts here============================================
        optimizer = Adam(self.model.parameters(), lr=self.lr) if not self.use_sgd else SGD(self.model.parameters(), lr=self.lr)
        scheduler = PolyLRScheduler(optimizer, self.lr, epochs)
           
        loss_function = my_import(self.loss_function) if self.loss_function != 'dice' else \
        DiceLoss()
        
        validation_score_data = []

        best_val_score = self.best_score
        
        for epoch in range(current_epoch, epochs):
            self._print(f'Epoch {epoch+1}/{epochs} on rank {self.device}.', f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")

            start = time.time()
            self.model.train()
            train_dataloader.sampler.set_epoch(current_epoch)

            loss_vals = []
            for batch in train_dataloader:
                loss_vals.append(self._trainstep(batch, loss_function, optimizer=optimizer).item())

            scheduler.step(epoch)
            if self.rank == 0:
                total = time.time() - start
                self._print(f"\nEpoch took {total} seconds.\n")
                self._print('Epoch train loss:', np.array(loss_vals).mean())
            
        #============================================Train done, start validation============================================

            self.model.eval()
            
            if self.rank == 0:
                self._print("\nRunning inference....")

            results = []
            loss_results = []
            for batch in validation_dataloader:
                with torch.no_grad():
                    score, loss = self._runvalidation(batch, loss_function)
                    results.append(score.item())
                    loss_results.append(loss.item())

            score = np.array(results).mean()
            loss = np.array(loss_results).mean()

            validation_score_data.append(score)
            if self.rank == 0:
                self._print(f'Mean dice: {score}')
                self._print(f'Validation loss: {loss}')
            last_score = score

        #============================================Done validation, save if applicable============================================
            if(last_score >= best_val_score):
                best_val_score = last_score
                if self.rank == 0:
                    self._print("New best validation score!")
                    if self.save_best:
                        self._print("Saving model progress.")
                        thread = threading.Thread(target=ResumeTrainingHelper.save_progress, args=(
                            self.model.module.state_dict(),
                            self.model.module.__class__, 
                            self.dataset_path, 
                            epoch,
                            best_val_score
                        ))
                        thread.start()
            
            if self.rank == 0:
                self._print("--------------------------------------------------------------") 
    
    def __preparebatch__(self, batch)-> torch.Tensor:
        inputs = batch['img']
        labels = batch['seg']

        if self.treat_2d:
            inputs = Trainer.__3dto2d__(inputs)
            labels = Trainer.__3dto2d__(labels)

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        return inputs, labels
    
    def initialize_logging(self, out:str):
        print(f"Logging to {out}")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=out,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level='DEBUG')
    def _print(self, *args):
        for arg in args:
            logging.info(arg)
        
    @staticmethod
    def __3dto2d__(data:torch.Tensor)->torch.Tensor:
        #Not batched data yet, so remove channels (first)
        assert data.shape[1] == 1, 'Cannot convert to 2D if there are multiple channels.'
        return data.reshape(shape=(data.shape[0], data.shape[2], data.shape[3], data.shape[4]))

def __fillparser__(parser:argparse.ArgumentParser)->None:
    parser.add_argument('-d', type=str, required=True, help="The absolute path to the dataset that you want to train.")
    parser.add_argument('-m', type=str, required=True, help="Either the name of your model class stored in src/models, "
                        + "or the path to a json file describing the model.")
    
    parser.add_argument('-log', type=str, required=True, help="Path for log file.")
    
    parser.add_argument('-folds', type=int, required=True, nargs='+', help="A list of folds to train. 0 based")
    
    parser.add_argument('--_2d', action='store_true', help="2D training? Make sure your model is 2D!")
    parser.add_argument('--resume', action='store_true', help="Resume your last training session.")
    parser.add_argument('--store', action='store_true', help="Save your preogress.")
    parser.add_argument('--amp', action='store_true', help="Enable automatic mixed precision.")
    parser.add_argument('--cpu', action='store_true', help="Run trianing on cpu.")
    parser.add_argument('--sgd', action='store_true', help="Use SGD optimizer algorithm.")


def get_dataset(patch_size:tuple, is_validation:bool, dataframe:pd.DataFrame, dataset_path:str):
    augs = get_validation_transformations(patch_shape=patch_size) \
    if is_validation else \
    get_train_augmentations(patch_shape=patch_size)
    num_workers = 8
    return get_dataset_from_dataframe(dataframe, augs, dataset_path, num_workers=num_workers)

def ddp_setup(rank: int, world_size: int, cpu:bool):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend=("nccl" if not cpu else 'gloo'), rank=rank, world_size=world_size)
    if not cpu:
        torch.cuda.set_device(rank)    

def main(rank, world_size, args, fold):
    ddp_setup(rank, world_size=world_size, cpu=args.cpu)
    t = Trainer(
        args.d,
        fold,
        rank,
        treat_2d=args._2d,
        batch_size=2,
        mixed_precision=args.amp,
        log = args.log
    )
    t.train_entry(
        epochs=100,
        model=args.m,
        save_best=args.store,
        resume_training=args.resume
    )
    if not args.cpu:
        destroy_process_group()

if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    __fillparser__(parser)
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count() if not args.cpu else 1

    folds_manager = FoldsManager(args.d)
    #If folds is -1, run all folds.
    for fold_id in args.folds:
        fold = folds_manager.getfold(fold_id)

        mp.spawn(
            main,
            args=(world_size, args, fold),
            nprocs=world_size,
        )