import multiprocessing as mp
import numpy as np
import queue
import time
import glob
import torch
from torch.utils.data import Dataset
import cv2


"""NOTE in the documentation transforms and augmented are used interchangeably"""

"""
Weed and Crop Dataset
    The dataset responsible for loading and transforming images 
    and masks for segmentation tasks

Attributes
----------------------------------------------------------------
    image_dir : np.array
        a numpy array holding all the path of the images
        
    mask_dir : np.array
        a numpy array holding all the path of the masks
    
    epochs : int
        the number of epochs the model will be trained with
    
    transform : 
        the function responsible for augmenting the image and mask
    
    num_processes : int
        the number of processes that the dataset will use
        
    path_queue : JoinableQueue
        the queue that will hold all the paths to the individual images and mask
        the item in the queue is stored as a tuple with the format (image_path, mask_path)
        it will hold epoch * images amount of item
    
    image_mask_queue : JoinableQueue
        the queue that will hold all the augmented images and mask
        the item in the queue is stored as a tuple with the format (augmented_image, augmented_mask)
        it will hold a maximum (epoch * images) + num_processes amount of item
        
    command_queue : JoinableQueue
        the queue that will determines when the transformation processes will end
        the item in the queue will only ever be None
        and it will hold a maximum num_processes amount of item
        
    read_transform_processes : List[process]
        the list that holds all the processes that will read and transform the images and mask
        
Methods
 ----------------------------------------------------------------
    __init__ : None
        store the image and mask directory, number of training epochs
        transformation function and the number of processes
        it will also define the joinable queues and define the read transform processes
        
        parameters:
        
            image_dir : str
                the absolute path to the directory containing the images
                
            mask_dir : str
                the absolute path to the directory containing the masks
                
            epochs : int, optional
                the number of epochs the model will be trained with
                default is 1
                
            transform : optional
                the transformation function that will augment the images and masks
                default is None
                
            num_processes : int, optional
                the number of processes
                default is 1
    
    __populate_path_queue__ : None
        populate the path_queue with the image and mask path in a random order
        
        parameters:
            None
        
    __read_transform_image_mask__ : None
        read the image and mask and augment them. The result will be enqueued onto image_mask_queue
        
        parameters:
            path_queue : JoinableQueue
                the queue containing all the paths
            
            image_mask_queue : JoinableQueue
                the output queue holding all transformed images and masks
            
            command_queue : JoinableQueue
                the queue that will define when this method will terminate
            
            transform : optional
                the transformation function that will be applied to the images and masks
                default is None
                
    start : None
        start the processes
        
        parameters:
            None
        
    Join : None
        join the processes and queue
        
        parameters:
            None
     
     __len__ : int
        return the length of the dataset
        
        parameters:
            None
    
    __getitem__ : tensor, tensor
        return the image and mask that is in front of image_mask_queue
        
        parameters:
            index : int
                following torch Dataset structure this function must have an index parameter
                however it will be ignored completely
"""


class WeedAndCropDataset(Dataset):

    def __init__(self, image_dir,
                 mask_dir,
                 epochs=1,
                 transform=None,
                 num_processes=1):
        # storing parameters
        self.image_dir = np.array(glob.glob(f'{image_dir}/*.png'))
        self.mask_dir = np.array(glob.glob(f'{mask_dir}/*.png'))
        self.epochs = epochs
        self.transform = transform
        self.num_processes = num_processes

        # defining the joinable queues
        self.path_queue = mp.JoinableQueue()
        self.image_mask_queue = mp.JoinableQueue()
        self.command_queue = mp.JoinableQueue()

        # defining the processes
        self.read_transform_processes = []

        for _ in range(num_processes):
            proc = mp.Process(target=WeedAndCropDataset.__read_transform_image_mask__,
                              args=(self.path_queue,
                                    self.image_mask_queue,
                                    self.command_queue,
                                    self.transform))
            self.read_transform_processes.append(proc)

    def __populate_path_queue__(self):
        for i in range(self.epochs):
            # shuffle the path array, so we can still get random order
            shuffler = np.random.permutation(len(self.image_dir))
            self.image_dir = self.image_dir[shuffler]
            self.mask_dir = self.mask_dir[shuffler]

            for j in range(len(self.image_dir)):
                self.path_queue.put((self.image_dir[j], self.mask_dir[j]))

        # adding sentinel values so that the consumer can terminate
        for i in range(self.num_processes):
            self.path_queue.put((None, None))

    """
    Consumer process of __populate_path_queue__
    Producer process to __getitem__
    """

    @staticmethod
    def __read_transform_image_mask__(path_queue: mp.JoinableQueue,
                                      image_mask_queue: mp.JoinableQueue,
                                      command_queue: mp.JoinableQueue,
                                      transform=None):

        while True:
            image_path, mask_path = path_queue.get()

            # sentinel value is read, time to terminate
            if image_path is None:
                path_queue.task_done()
                image_mask_queue.put((None, None))

                break

            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # converting the type of number from int to float and turn the pixels into the range [0,1]
            image = np.array(image, dtype=np.float32) / 255.0
            mask = np.array(mask, dtype=np.float32) / 255.0

            # applying the transformations
            if transform is not None:
                augmented = transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            # converting the image and mask into tensors
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask).unsqueeze(0)

            # putting the image and mask into a queue
            image_mask_queue.put((image, mask))

            # telling the queue the task is done
            path_queue.task_done()

        while True:
            try:
                out = command_queue.get()
                if out is None:
                    command_queue.task_done()
                    break
            except queue.Empty:
                time.sleep(0.5)
                continue

    """
    Populate queue path and initialize the processes
    """

    def start(self):
        self.__populate_path_queue__()

        for process in self.read_transform_processes:
            process.start()

    """
    Join the processes and terminates them
    """

    def join(self):

        for process in self.read_transform_processes:
            process.join()

        self.image_mask_queue.join()

    """DEFAULT METHODS FOR DATASET"""

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        try:
            image, mask = self.image_mask_queue.get()
            self.image_mask_queue.task_done()

            if image is None:
                self.command_queue.put(None)
                return None, None

            return image, mask
        except queue.Empty:
            time.sleep(0.2)  # wait a bit before trying again
            return self.__getitem__(index)
