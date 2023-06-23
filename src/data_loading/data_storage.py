import SimpleITK as sitk
import pandas as pd
import numpy as np
import torchio as tio
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

class SegmentObject:
    """
    Loads the segmentation and the original image into simple stk objects.
    To get the data as numpy arrays call the get array methods.
    """
    def __loaddata__(self, segmentation_only:bool = False, image_only:bool = False)->None:
        """
        Reads data to memory.
        """
        assert not (segmentation_only and image_only), 'Parameter segmentation_only and image_only are mutually exclusive.'
        assert self.data_paths['Data Available'], 'Data does not exist for this point. Ensure it is downloaded. (the Data Availale column is False)'
        segmentation = sitk.ReadImage(self.data_paths['Segmentation'])
        
        if not segmentation_only:
            image = sitk.ReadImage(self.data_paths['Image'])
            self.image_stk = image

        self.segmentation_stk = segmentation

    def __init__(self, data_paths:pd.Series, load:bool = False)->None:
        """
        Loads data from a given row of a data frame.
        To load the data right away set the load parameter to True, but this is unececarry, and the data can be loaded when it is requested.
        This object also stores a TorchIO Subject. This ibject contains a path o the map, as well as image.
        The tio.subject object has the dict entry 'class' which can be used to link that object back to this object. 
        """
        self.data_paths = data_paths
        self.case_id = data_paths['Case']
        self.segmentation_stk = None
        self.image_stk = None

        self.monai_dict = {'img':data_paths['Image'], 'seg':data_paths['Segmentation']}
        if load:
            self.__loaddata__()

    def get_segmentation_array(self, store_data:bool=False, view:bool=False, segmentation_only:bool = False)->np.array:
        """
        Returns the array representation of the segmentation. Stores in the object if store_data is True.
        """
        #sitk object is (width, height, depth)/(x, y, z). np array is (depth, height, width)/(z, x, y)
        if self.segmentation_stk == None:
            self.__loaddata__(segmentation_only = segmentation_only)
        
        seg = copy.deepcopy(self.segmentation_stk)

        if not store_data:
            self.image_stk = None
            self.segmentation_stk = None

        return sitk.GetArrayFromImage(seg)
    
    def get_image_array(self, store_data:bool=False, image_only:bool = False)->np.array:
        """
        Returns the array representation of the imaging. Stores in the object if store_data is True.
        """
        if self.segmentation_stk == None:
            self.__loaddata__(image_only = image_only)
        
        im = copy.deepcopy(self.image_stk)
        if not store_data:
            self.image_stk = None
            self.segmentation_stk = None

        return sitk.GetArrayFromImage(im)

    def show(self, axis:int=0, first_frame:int = 0, frames:int = 3, segmentation_overlap:bool = True)->None:
        """
        Displays n frames. Frames are displayed as follows: {f, f+1, f+2 .... f+n} with an overlapping segmentation if applicable.
        Calling this will load the data into memory if not already there, and then clear it again after
        """
        delete_after = ((self.segmentation_stk == None) or (self.image_stk == None))

        assert 0 <= axis <= 2, f"Invalid axis {axis} for length 3."
        assert frames <= 10, "You can only show up to 10 frames."

        _, plts = plt.subplots(1, frames, figsize=(15, 15))
        image =  self.get_image_array()
        mask = self.get_segmentation_array()

        for frame in range(first_frame, frames+first_frame):
            display_a = None
            display_b = None
            if axis == 0:
                display_a = image[frame,:,:]
                display_b = mask[frame,:,:]
            elif axis == 1:
                display_a = image[:,frame,:]
                display_b = mask[:,frame,:]
            else:
                display_a = image[:,:,frame]
                display_b = mask[:,:,frame]

            plts[frame-first_frame].axis('off')
            plts[frame-first_frame].imshow(display_a.squeeze(), cmap='gray')
            if segmentation_overlap:
                plts[frame-first_frame].imshow(display_b.squeeze(), alpha=0.25)
        
        if delete_after:
            self.image_stk = None
            self.segmentation_stk = None


    @staticmethod
    def points_from_df(df:pd.DataFrame, load:bool = False, verbose = False)->list():
        """
        Given a pandas dataframe, returns a list of SegmentObjects. if load is true, automatically loads data into memory.
        """
        data_points = []
        unavailable = 0
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):

            if row['Data Available']:
                point = SegmentObject(row, load=load)
                data_points.append(point)
                df.loc[i, 'Object Address'] = point
            else:            
                unavailable += 1

        if unavailable > 0 and verbose:
            print(f'{unavailable} data points are not downloaded. If this is not the case, re-run the case_to_series method.')
        elif verbose:
            print("All points loaded succesfully!")
        
        return data_points
    
if __name__ == '__main__':
    pass