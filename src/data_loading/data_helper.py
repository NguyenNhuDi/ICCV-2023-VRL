import sys
sys.path.append('/home/student/andrew/Documents/Seg3D/src')

import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import copy
from tqdm import tqdm
import os
from utils.helpers import get_case_from_path
from .data_storage import SegmentObject
from monai.data import Dataset, CacheDataset, PersistentDataset

def kits23_to_dataset(dest:str, data:pd.DataFrame = None)->None:
    """
    Move data to new dataset form.
    """
    assert type(data) != None, "Pass data argument next time..."
    import shutil, os
    try:
        dest = f'{dest}/data'
        os.mkdir(dest)
    except:
        pass
    available_data = data.loc[data['Data Available'] == True]
    if(len(available_data) < len(data)):
        print(f"Warning, there are {len(data) - len(available_data)} data points that were not loaded!\
              \n(The dataframe says data available is false... probably didn't run kits23_download_data or specified kiney only when that data doesnt exist.).")

    #For each datapoint, we will copy the data from the kits23 format into our own similar file structure.
    for _, row in tqdm(available_data.iterrows(), total=len(available_data)):
        image_path, segmentation_path = row['Image'], row['Segmentation']
        case = get_case_from_path(image_path)
        #Make the case folder. If it fails, likely already exists.
        try:
            os.makedirs(f'{dest}/{case}')
        except:
            pass
        #Keeps the file name and metadata.
        shutil.copy2(image_path, f'{dest}/{case}')
        shutil.copy2(segmentation_path, f'{dest}/{case}')


def get_dataset_from_dataframe(data:pd.DataFrame, transforms, dataset_path:str, num_workers:int=4)->CacheDataset:
    """
    Given the main dataframe, get a TorchIO dataset. Each point in the dataset is a tio.Subject.
    Subjects contain the image and the mask. Transformation apply to both, or only to image when appropriate.

    This dataset can be used with the torch.utils.data.Dataloader
    """
    assert 'Object Address' in data.columns, 'No objects in dataframe, smh'
    try:
        os.mkdir(f'{dataset_path}/cache')
    except:
        pass

    data = [d.monai_dict for d in data['Object Address']]
    #return CacheDataset(data, transforms, num_workers=num_workers, cache_rate=0.75) #If none, uses os.cpu_count()
    return PersistentDataset(data, cache_dir = f'{dataset_path}/cache', transform=transforms)

def get_case_series_kits(folder_path:str, kidney_only:bool=False)->pd.Series:
    """
    Given the path to a case folder, returns a pandas series with: [Case id, image path, segmentation path, boolean: imaging data exists]
    If kidney only is false, looks for folder_path/segmentation.nii.gz, otherwise folder_path/segmentation_kidney.nii.gz
    """
    segmentation = f'{folder_path}/segmentation.nii.gz' if not kidney_only else f'{folder_path}/segmentation_kidney.nii.gz'
    imaging = f'{folder_path}/imaging.nii.gz'
    data_exists = os.path.exists(imaging) and os.path.exists(segmentation)

    folder_path = folder_path.replace('\\', '/')
    case = folder_path.split('/')[-1]
    assert 'case' in case, f'The path is not as expected. Should be ./kits23/dataset/case_xxxxx got {folder_path}'
    case = case.split('_')[1]
    #TODO assign test/train
    return pd.Series([case, imaging, segmentation, data_exists], ['Case', 'Image', 'Segmentation', 'Data Available'])

def __get_case_series__(folder_path:str):
    """
    Given the path to a dataset folder, returns a pandas series with: [Case id, image path, segmentation path, boolean: imaging data exists]
    Looks for folder_path/segmentation.nii.gz, otherwise folder_path/segmentation_kidney.nii.gz
    """
    segmentation = f'{folder_path}/segmentation.nii.gz'
    if not os.path.exists(segmentation):
        segmentation = f'{folder_path}/segmentation_kidney.nii.gz'
        assert os.path.exists(segmentation)

    imaging = f'{folder_path}/imaging.nii.gz'
    data_exists = os.path.exists(imaging) and os.path.exists(segmentation)

    folder_path = folder_path.replace('\\', '/')
    case = folder_path.split('/')[-1]
    int(case)
    #TODO assign test/train
    return pd.Series([case, imaging, segmentation, data_exists], ['Case', 'Image', 'Segmentation', 'Data Available'])

def get_dataframe_fram_dataset(dataset_path:str)->pd.DataFrame:
    """
    Given a dataset, returns a dataframe containing all data and paths to it.
    """
    import glob

    cases = glob.glob(f'{dataset_path}/data/*')
    cases = [path for path in cases if 'json' not in path]
    all_series = [__get_case_series__(path) for path in cases]
    df = pd.DataFrame(all_series)
    SegmentObject.points_from_df(df)

    return df