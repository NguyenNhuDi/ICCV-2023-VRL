import math
import multiprocessing as mp
import numpy as np
import queue
import time
import glob
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A

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
            index_queue : JoinableQueue
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


class WeedAndCropDataset:

    def __init__(self, image_dir,
                 mask_dir,
                 batch_size=1,
                 epochs=1,
                 num_processes=1,
                 transform=None,
                 ):

        # storing parameters
        self.image_dir = np.array(glob.glob(f'{image_dir}/*.png'))
        self.mask_dir = np.array(glob.glob(f'{mask_dir}/*.png'))
        self.epochs = epochs
        self.transform = transform
        self.num_processes = num_processes
        self.batch_size = batch_size

        # defining the joinable queues
        self.index_queue = mp.JoinableQueue()
        self.image_mask_queue = mp.JoinableQueue()
        self.command_queue = mp.JoinableQueue()

        # storing indexes to the path array
        self.index_arr = np.array([i for i in range(len(self.image_dir))])

        self.total_size = self.epochs * self.__len__()

        # defining the processes
        self.read_transform_processes = []

        for _ in range(num_processes):
            proc = mp.Process(target=WeedAndCropDataset.__batch_image_mask__,
                              args=(self.image_dir,
                                    self.mask_dir,
                                    self.index_queue,
                                    self.image_mask_queue,
                                    self.command_queue,
                                    self.transform))
            self.read_transform_processes.append(proc)

        # counter to tell when the processes terminate
        self.accessed = 0

    def __populate_index_queue__(self):
        # Does the first epoch - 1 times

        index_batch = []
        index_counter = 0
        total_counter = 0
        batch_counter = 0

        while True:
            if index_counter == len(self.index_arr):
                index_counter = 0
                shuffler = np.random.permutation(len(self.index_arr))
                self.index_arr = self.index_arr[shuffler]

            if total_counter == self.total_size:
                if len(index_batch) > 0:
                    self.index_queue.put(index_batch)
                break

            index_batch.append(self.index_arr[index_counter])
            index_counter += 1
            total_counter += 1
            batch_counter += 1

            if batch_counter == self.batch_size:
                self.index_queue.put(index_batch)
                index_batch = []
                batch_counter = 0

        for _ in range(self.num_processes):
            self.index_queue.put(None)

    @staticmethod
    def __read_transform_image_mask__(image_path, mask_path, transform=None):
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
        image = torch.from_numpy(image).permute(2, 1, 0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

    """
    Consumer process of __populate_index_queue__
    Producer process to __getitem__
    """

    @staticmethod
    def __batch_image_mask__(image_paths: np.arry,
                             mask_paths: np.array,
                             index_queue: mp.JoinableQueue,
                             image_mask_queue: mp.JoinableQueue,
                             command_queue: mp.JoinableQueue,
                             transform=None):
        while True:
            indexes = index_queue.get()
            index_queue.task_done()

            if indexes is None:
                break

            image_batch = []
            mask_batch = []
            for index in indexes:
                image = image_paths[index]
                mask = mask_paths[index]

                image, mask = WeedAndCropDataset.__read_transform_image_mask__(image, mask, transform)

                image_batch.append(image)
                mask_batch.append(mask)

            image_batch = torch.stack(image_batch, dim=0)
            mask_batch = torch.stack(mask_batch, dim=0)
            image_mask_queue.put((image_batch, mask_batch))

            # Waiting for get_item to be finished with the queue
            while True:
                try:
                    sent_val = command_queue.get()
                    if sent_val is None:
                        command_queue.task_done()
                        break
                except queue.Empty:
                    time.sleep(0.5)
                    continue

    """
    Populate queue path and initialize the processes
    """

    def start(self):
        self.__populate_index_queue__()

        # for process in self.read_transform_processes:
        #     process.start()

    """
    Join the processes and terminates them
    """

    def join(self):

        for process in self.read_transform_processes:
            process.join()

        self.image_mask_queue.join()

    """
                            SINGLE THREADED BELOW
    ________________________________________________________________________
    """

    # create batch method
    def __len__(self):
        return len(self.image_dir)

    def get_item(self):
        try:
            image, mask = self.image_mask_queue.get()
            self.image_mask_queue.task_done()
            self.accessed += 1

            # if the none counter is the same amount of processes this means that all processes eof is reached
            # deploy the None into command queue to terminate them
            # this is essential in stopping NO FILE found error

            if self.accessed == self.max_queue_size:
                for j in range(self.num_processes):
                    self.command_queue.put(None)
            return image, mask

        except queue.Empty:
            time.sleep(0.01)
            return self.get_item()


if __name__ == '__main__':
    # image_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\SMH SMH\image'
    # mask_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\SMH SMH\mask'

    image_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\PhenoBench\train\images'
    mask_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\PhenoBench\train\leaf_instances'

    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.HueSaturationValue()
    ])

    epochs = 10
    num_processes = 6
    batch_size = 32

    test_dataset = WeedAndCropDataset(image_path,
                                      mask_path,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      num_processes=num_processes,
                                      transform=transform)
    test_dataset.start()

    start = time.time_ns()

    while test_dataset.index_queue.qsize() > 0:
        out = test_dataset.index_queue.get()
        if out is None:
            print(None)
        else:
            print(out)
            print(len(out))

    # for i in range(test_dataset.max_queue_size):
    #     image, mask = test_dataset.get_item()
    #     print(f'Iteration: {i}, shape: {image.shape}, queue size: {test_dataset.image_mask_queue.qsize()}')
    #     # MUMBO JUMBO CODE JUST TESTING THE SPEED OF HOW FAST WE CAN GET IMAGE

    end = time.time_ns()

    test_dataset.join()

    print(end - start)
