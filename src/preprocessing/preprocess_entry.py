import os
import sys
sys.path.append('/home/andrewheschl/Documents/3DSegmentation/src')

from fingerprint import FingerprintExtractor
from data_loading.data_helper import kits23_to_dataset, \
                                        get_case_series_kits, get_dataframe_fram_dataset

from utils.folds_manager import FoldsManager

from data_loading.data_storage import SegmentObject

import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def transfer_kits_to_dataset(dataset:str, data:pd.DataFrame) -> None:
    """
    Given a dataframe of kits paths, move to new dataset.
    """
    kits23_to_dataset(dataset, data=data)

def transfer_kits() -> None:
    """
    Given the path to a kits23 dataset, copy the data to a new structure.
    """
    data_path = str(input('Enter the path to the kits23 dataset ex. ./kits23/dataset: '))
    #Ensure the entered path exists
    assert os.path.isdir(data_path), f"Path '{data_path}' does not exist"
    
    #Ask for the target dataset path
    dataset_path = str(input('Enter a target dataset path: '))

    #Start conversion.
    kidney_only = str(input('Kidney only? (Y/N): ')) == 'Y' #Did you run the previous tool to generate kidney only masks?
    print(seperator)

    print("Working with kidney only masks." if kidney_only else "Working on full mask with multiple labels.")
    
    cases = glob.glob(f'{data_path}/*')
    #Get dataframe series from each case
    all_series = [get_case_series_kits(path, kidney_only=kidney_only) for path in cases]
    data = pd.DataFrame(all_series)
    #Move the goods
    transfer_kits_to_dataset(dataset_path, data)
    #We need the entered path later
    return dataset_path

def generate_folds(dataset_path:str):
    k = int(input('How many splits? '))
    FoldsManager.generate_k_folds(dataframe, dataset_path, k=k)

def reshape_data(df:pd.DataFrame, dataset_path:str, target_size:tuple):
    from scipy.ndimage import zoom

    left, right, top, bottom, front, back = 0, 0, 0, 0, 0, 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        #Pad evenly on either side if any dimension is too small.
        image_array = row['Object Address'].get_image_array()
        segmentation_array = row['Object Address'].get_segmentation_array()

        if image_array.shape[0]<target_size[0]:
            residual = target_size[0]-image_array.shape[0]
            left = residual//2
            right = residual//2
            if residual % 2 == 1:
                right+=1
        
        if image_array.shape[1]<target_size[1]:
            residual = target_size[1]-image_array.shape[1]
            top = residual//2
            bottom = residual//2
            if residual % 2 == 1:
                bottom+=1
            
        if image_array.shape[2]<target_size[2]:
            residual = target_size[2]-image_array.shape[2]
            front = residual//2
            back = residual//2
            if residual % 2 == 1:
                back+=1

        image_array = np.pad(image_array, ((left, right), (top, bottom), (front, back)), constant_values=0)
        segmentation_array = np.pad(segmentation_array, ((left, right), (top, bottom), (front, back)), constant_values=0)

        #If this is true, then the data is too large. Resize it.
        if image_array.shape != target_size:
            x = target_size[0]/image_array.shape[0] if image_array.shape[0] > target_size[0] else 1
            y = target_size[1]/image_array.shape[1] if image_array.shape[1] > target_size[1] else 1
            z = target_size[2]/image_array.shape[2] if image_array.shape[2] > target_size[2] else 1

            image_array = zoom(image_array, (x, y, z))
            segmentation_array = zoom(segmentation_array, (x,y,z))
            pass
        
        assert image_array.shape == target_size, f"Expected shape {target_size}, but got {image_array.shape}"
        assert segmentation_array.shape == target_size, f"Expected shape {target_size}, but got {segmentation_array.shape}"
        
        sitk.WriteImage(
            sitk.GetImageFromArray(image_array), 
            row['Image']
        )
        sitk.WriteImage(
            sitk.GetImageFromArray(segmentation_array), 
            row['Segmentation']
        )
        #Free memory
        row['Object Address'].image_stk = None
        row['Object Address'].segmentation_stk = None

def split_test(dataset_path:str, df:pd.DataFrame)->None:
    from sklearn.model_selection import train_test_split
    import shutil

    train_path = f'{dataset_path}/data'
    test_path = f'{dataset_path}/test'

    seed = 123
    _, test_df = train_test_split(df, train_size=0.8, random_state=seed)
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        case = row['Case']
        shutil.move(f'{train_path}/{case}', f'{test_path}/{case}')

def preprocess(df:pd.DataFrame, dataset_path:str, get_print:bool = False, split_into_test:bool=False, reshape:bool=True)->None:
    """
    Given paths to each item in a dataset, run preprocessing steps.
    """
    if 'Object Address' not in df.columns:
        #We need this...
        SegmentObject.points_from_df(df, load=False)
    
    print(seperator)

    if reshape:
        print("Reshaping data")
        reshape_data(dataset_path=dataset_path, df=df, target_size=(512, 512, 100))
    if split_into_test:
        print("Splitting into train/test at a 0.8 split.")
        split_test(dataset_path, df)
    #Fingerprint right now just includes the mean and std for normalization.
    if get_print:
        print("Generating fingerprint.")
        FingerprintExtractor(dataset_path, df).write()

if __name__ == '__main__':
    """
    This tool allows you to:
    a) Convert kits23 dataset to proper form
    b) Generate splits file for dataset
    c) Will add cropping and any other preprocessing to dataset
    """
    data = None
    seperator = "--------------------------------------------------------------------------------------------------------------------"
    dataset_path = None

    #First, see if user wants to generate dataset from kits
    if str(input('Do you want to transfer kits23 to new datset? (Y/N): ')) == 'Y':
        dataset_path = transfer_kits()
    else:
        dataset_path = str(input('Enter the path of your dataset: '))
    
    assert os.path.isdir(dataset_path), f'Dataset {dataset_path} does not exist. smh'
    print(seperator)
    print(f'Loading dataset {dataset_path}.\n')

    #Gets the data from the dataset.
    dataframe = get_dataframe_fram_dataset(dataset_path)
    print(dataframe.head())
    print(seperator)
    #We can now generate a file to define different folds.
    if str(input("Do you want to generate a splits file (this will overwite the old one)? (Y/N)")) == "Y":
        generate_folds(dataset_path=dataset_path)
    #This is where we can start to preprocess. This right now just includes fingerprint, need to encorperate cropping.
    if str(input("Do you want to preprocess? (Y/N) ")) == "Y":
        reshape = str(input('Do you want to reshape your data?(Y/N) ')) == 'Y'
        split_set = str(input('Do you want to split into train/test?(Y/N) ')) == 'Y'
        get_print = str(input('Do you want to generate a new fingerprint?(Y/N) ')) == 'Y'
        preprocess(dataframe, dataset_path, get_print=get_print, split_into_test=split_set, reshape=reshape)
    
    print('This is all the preprocessing for now..')
    

