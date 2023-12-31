import math
import multiprocessing as mp
import numpy as np
import queue
import time
import torch

"""NOTE in the documentation transforms and augmented are used interchangeably"""

"""
DSAL (Dataset and loader)
    This clas will transform and batch images/labels/labels for machine learning tasks
    To use this class, make sure to define a function that will read and transform your images/labels
    this function should have the (images, label_obj, transform) as parameters

    If DSAL is used for csv label reading, simply pass in the absolute path to the label csv into the
    yml parameter slot


Attributes
----------------------------------------------------------------
    images : np.array
        a numpy array holding all the path of the images

    yml : np.array
        a numpy array holding all the path of the labels

    read_and_transform_function : function
        this function will be used to read and transform the images/labels

    epochs : int
        the number of epochs the model will be trained with

    transform : 
        the function responsible for augmenting the image and label

    num_processes : int
        the number of processes that the dataset will use

    batch_size : int
        the size of the batches

    index_queue : JoinableQueue
        the queue that will hold all the indexes to the different paths in batches

    image_label_queue : JoinableQueue
        the queue that will hold all the augmented images and label
        the item in the queue is stored as a tuple with the format (image_batch, label_batch)

    command_queue : JoinableQueue
        the queue that will determines when the transformation processes will end
        the item in the queue will only ever be None
        and it will hold a maximum num_processes amount of item

    index_arr : np.array
        an array that holds the indexes to the paths, this array will be shuffled to 
        simulate randomness

    read_transform_processes : List[process]
        the list that holds all the processes that will read and transform the images and label

    accessed : int
        counter that is used in get_item to tell when image_label_queue is empty

    total_size : int
        the total amount of images and labels that will be processed (epoch * number of images/labels)

    num_batches : int
        total number of batches


Methods
 ----------------------------------------------------------------
    __init__ : None
        store the image and label directory, number of training epochs
        transformation function and the number of processes
        it will also define the joinable queues and define the read transform processes

        parameters:

            images : str
                the absolute path to the directory containing the images

            yml : str
                the absolute path to the directory containing the labels

            read_and_transform_function : function
                the function that the user wrote to read a image and label path and
                apply transformations to it. The parameters should be in the form
                (images, yml, transform). It should also return the transformed
                image and label

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
                the transformation function that will augment the images and labels
                default is None



    __populate_index_queue__ : None
        populate the index queue with bathes of indexes in random order

        parameters:
            None

    __read_transform_image_label__ : tensor, tensor
        read the image and label and augment them. 
        the augmented image and label will be returned as tensors

        parameters:
            image_path : str
                the absolute path to the image that is being augmented
            label_path : str
                the absolute path to the label that is being augmented
            transform : function
                the transformation function that will transform the label and image

    __batch_image_label__ : None
        put images and labels in a batch and enqueue it into image_label_queue

        parameters:
            images_arr : np.array
                the array holding all the images

            label_paths : np.array
                the array holding all the label paths

            index_queue : mp.JoinableQueue
                the queue holding all the indexes

            image_label_queue : mp.JoinableQueue
                the output queue where the augmented image batch and label batch 
                will be enqueued into

            command_queue : mp.JoinableQueue
                the queue that will determine when the process running this method
                will terminate

            transform : function
                the transformation function that will augment the image and label

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
        return the image and label batch that is in front of image_label_queue

        parameters:
            None 
"""


class DSAL:

    def __init__(self, images,
                 yml,
                 read_and_transform_function,
                 cut_mix_function,
                 batch_size=1,
                 epochs=1,
                 num_processes=1,
                 max_queue_size=50,
                 transform=None,
                 mean=None,
                 std=None):

        assert batch_size >= 1, 'The batch size entered is <= 0'
        assert epochs >= 1, 'The epochs entered is <= 0'
        assert num_processes >= 1, 'The number of processes entered is <= 0'

        # storing parameters
        self.images = images

        # check to see if this is a path to labels or label csv
        self.yml = list(yml)

        self.read_and_transform_function = read_and_transform_function
        self.cut_mix_function = cut_mix_function
        self.epochs = epochs
        self.transform = transform
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.mean = mean
        self.std = std

        # defining the joinable queues
        self.index_queue = mp.JoinableQueue()
        self.image_label_queue = mp.JoinableQueue(max_queue_size)
        self.command_queue = mp.JoinableQueue()

        # storing indexes to the path array
        self.index_arr = []
        for i in range(len(self.images)):

            if cut_mix_function is not None:
                self.index_arr.append((i, True))

            self.index_arr.append((i, False))


        self.index_arr = np.array(self.index_arr)

        self.total_size = self.epochs * self.__len__()

        # defining the processes
        self.read_transform_processes = []

        for _ in range(num_processes):
            proc = mp.Process(target=self.__batch_image_label__,
                              args=(self.read_and_transform_function,
                                    self.cut_mix_function,
                                    self.images,
                                    yml,
                                    self.index_queue,
                                    self.image_label_queue,
                                    self.command_queue,
                                    self.transform,
                                    mean,
                                    std))
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

    """
    Consumer process of __populate_index_queue__
    Producer process to __getitem__
    """

    @staticmethod
    def __batch_image_label__(read_and_transform_function,
                              cut_mix_function,
                             images_arr: np.array,
                             yml,
                             index_queue: mp.JoinableQueue,
                             image_label_queue: mp.JoinableQueue,
                             command_queue: mp.JoinableQueue,
                             transform=None,
                             mean=None,
                             std=None):
        while True:
            indexes = index_queue.get()
            index_queue.task_done()

            if indexes is None:
                break

            image_batch = []
            label_batch = []

            for item in indexes:
                index, cut_mix = item


                image, image_name = images_arr[index]

                labels = yml[image_name]

                # Transforming it normally
                if not cut_mix:
                    image, label = read_and_transform_function(image, labels, transform, mean, std)
                    image_batch.append(image)
                    label_batch.append(label)

                # Cut mixing it :D
                else:
                    cm_image, cm_label = cut_mix_function(image, image_name, labels, transform, mean, std)
                    image_batch.append(cm_image)
                    label_batch.append(cm_label)





            image_batch = torch.stack(image_batch, dim=0)
            label_batch = torch.stack(label_batch, dim=0)

            image_label_queue.put((image_batch, label_batch))
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

        self.image_label_queue.join()

    """
                            SINGLE THREADED BELOW
    ________________________________________________________________________
    """

    # create batch method
    def __len__(self):
        return len(self.images)

    def get_item(self):
        try:
            image, label = self.image_label_queue.get()
            self.image_label_queue.task_done()
            self.accessed += 1

            # if the none counter is the same amount of processes this means that all processes eof is reached
            # deploy the None into command queue to terminate them
            # this is essential in stopping NO FILE found error
            if self.accessed == self.num_batches:
                for j in range(self.num_processes):
                    self.command_queue.put(None)
            return image, label

        except Exception as e:
            print(e, flush=True)


        # except queue.Empty:
        #     time.sleep(0.01)
        #     return self.get_item()

