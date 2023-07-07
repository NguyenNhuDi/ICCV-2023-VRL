import math
import multiprocessing as mp
import numpy as np
import queue
import time
import glob
import torch
import cv2
import albumentations as A

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


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
        
    batch_size : int
        the size of the batches
        
    index_queue : JoinableQueue
        the queue that will hold all the indexes to the different paths in batches
    
    image_mask_queue : JoinableQueue
        the queue that will hold all the augmented images and mask
        the item in the queue is stored as a tuple with the format (image_batch, mask_batch)
        
    command_queue : JoinableQueue
        the queue that will determines when the transformation processes will end
        the item in the queue will only ever be None
        and it will hold a maximum num_processes amount of item
        
    index_arr : np.array
        an array that holds the indexes to the paths, this array will be shuffled to 
        simulate randomness
        
    read_transform_processes : List[process]
        the list that holds all the processes that will read and transform the images and mask
        
    accessed : int
        counter that is used in get_item to tell when image_mask_queue is empty
        
    total_size : int
        the total amount of images and masks that will be processed (epoch * number of images/masks)
    
    num_batches : int
        total number of batches
    
        
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
            
            batch_size : int, optional
                the size of the batches
                default is 1
                
            epochs : int, optional
                the number of epochs the model will be trained with
                default is 1
                
            num_processes : int, optional
                the number of processes
                default is 1
                
            transform : optional
                the transformation function that will augment the images and masks
                default is None
                

    
    __populate_index_queue__ : None
        populate the index queue with bathes of indexes in random order
        
        parameters:
            None
        
    __read_transform_image_mask__ : tensor, tensor
        read the image and mask and augment them. 
        the augmented image and mask will be returned as tensors
        
        parameters:
            image_path : str
                the absolute path to the image that is being augmented
            mask_path : str
                the absolute path to the mask that is being augmented
            transform : function
                the transformation function that will transform the mask and image
    
    __batch_image_mask__ : None
        put images and masks in a batch and enqueue it into image_mask_queue
        
        parameters:
            image_paths : np.array
                the array holding all the image paths
            
            mask_paths : np.array
                the array holding all the mask paths
            
            index_queue : mp.JoinableQueue
                the queue holding all the indexes
            
            image_mask_queue : mp.JoinableQueue
                the output queue where the augmented image batch and mask batch 
                will be enqueued into
                
            command_queue : mp.JoinableQueue
                the queue that will determine when the process running this method
                will terminate
            
            transform : function
                the transformation function that will augment the image and mask
                
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
    
    get_item : tensor, tensor
        return the image and mask batch that is in front of image_mask_queue
        
        parameters:
            None 
"""


class WeedAndCropDataset:

    def __init__(self, image_dir,
                 mask_dir,
                 batch_size=1,
                 epochs=1,
                 num_processes=1,
                 max_queue_size=50,
                 transform=None):

        assert batch_size >= 1, 'The batch size entered is <= 0'
        assert epochs >= 1, 'The epochs entered is <= 0'
        assert num_processes >= 1, 'The number of processes entered is <= 0'

        # storing parameters
        self.image_dir = np.array(glob.glob(f'{image_dir}/*.png'))
        self.mask_dir = np.array(glob.glob(f'{mask_dir}/*.png'))
        self.epochs = epochs
        self.transform = transform
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        # defining the joinable queues
        self.index_queue = mp.JoinableQueue()
        self.image_mask_queue = mp.JoinableQueue(max_queue_size)
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
                                    self.max_queue_size,
                                    self.transform))
            self.read_transform_processes.append(proc)

        # counter to tell when the processes terminate
        self.accessed = 0

        # variable to use when running training loop
        self.num_batches = math.ceil(self.total_size / self.batch_size)

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
    def __batch_image_mask__(image_paths: np.array,
                             mask_paths: np.array,
                             index_queue: mp.JoinableQueue,
                             image_mask_queue: mp.JoinableQueue,
                             command_queue: mp.JoinableQueue,
                             max_queue_size,
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

        for process in self.read_transform_processes:
            process.start()

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
            if self.accessed == self.num_batches:
                for j in range(self.num_processes):
                    self.command_queue.put(None)
            return image, mask

        except queue.Empty:
            time.sleep(0.01)
            return self.get_item()


class TorchTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = glob.glob(f'{image_dir}/*.png')
        self.mask_dir = glob.glob(f'{mask_dir}/*.png')
        self.transform = transform

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image = cv2.imread(self.image_dir[index], cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_dir[index], cv2.IMREAD_GRAYSCALE)

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

    epochs = 4
    num_processes = 6
    batch_size = 32

    test_dataset = WeedAndCropDataset(image_path,
                                      mask_path,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      num_processes=num_processes,
                                      max_queue_size=num_processes * 3,
                                      transform=transform)

    torch_dataset = TorchTestDataset(image_path, mask_path, transform=transform)
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, num_workers=num_processes)

    print('starting....')
    test_dataset.start()

    print("starting finished\nbegin testing...")

    start = time.time_ns()
    for i in tqdm(range(test_dataset.num_batches)):
        image, mask = test_dataset.get_item()
        # MUMBO JUMBO CODE JUST TESTING THE SPEED OF HOW FAST WE CAN GET IMAGE

    end = time.time_ns()

    test_dataset.join()

    print(end - start)

    start = time.time_ns()

    for i, data in enumerate(tqdm(torch_dataloader, 0)):
        image, mask = data
        # MUMBO JUMBO CODE JUST TESTING THE SPEED OF HOW FAST WE CAN GET IMAGE

    end = time.time_ns()
    print(end - start)


