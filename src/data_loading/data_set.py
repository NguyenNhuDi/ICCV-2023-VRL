import logging
import multiprocessing as mp
import queue
import time

import glob

import torch
import random
from torch.utils.data import Dataset
import cv2


class WeedAndCropDataset(Dataset):
    def __init__(self, image_dir,
                 mask_dir,
                 epochs,
                 transform=None,
                 num_processes=1):
        self.image_source = glob.glob(f'{image_dir}/*.png')
        self.mask_source = glob.glob(f'{mask_dir}/*.png')
        self.transform = transform
        self.num_processes = num_processes
        self.epochs = epochs

        # Queue creations
        self.path_queue = mp.JoinableQueue()
        self.image_queue = mp.JoinableQueue()
        self.mask_queue = mp.JoinableQueue()
        self.logger_queue = mp.JoinableQueue()

        # Starting the processes

        self.processes = []

        for _ in range(num_processes):
            proc = mp.Process(target=WeedAndCropDataset.__get_and_transform_image__,
                              args=(self.path_queue,
                                    self.image_queue,
                                    self.mask_queue,
                                    self.logger_queue,
                                    self.transform))

            proc.daemon = True
            self.processes.append(proc)

        self.logger = mp.Process(target=WeedAndCropDataset.__logger_process__,
                                 args=(self.logger_queue,))

    def __populate_path_queue__(self):
        # putting sentinel values to the path queue so that it can end
        for i in range(self.num_processes):
            self.path_queue.put((None, None))

        for i in range(self.epochs):
            random.shuffle(self.image_source)
            random.shuffle(self.mask_source)

            for j in range(len(self.image_source)):
                self.path_queue.put((self.image_source[i], self.mask_source[i]))

    @staticmethod
    def __logger_process__(logger_queue: mp.JoinableQueue):
        logging.basicConfig(filename='transform_error.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s',
                            force=True)
        logger = logging.getLogger()

        while True:
            try:
                message = logger_queue.get(1)
                if message is None:
                    break
                logger.error(message)
            except queue.Empty:
                time.sleep(1)  # Sleep for a while before trying again.
                print('Empty')
                continue
            else:
                logger_queue.task_done()

    @staticmethod
    def __get_and_transform_image__(path_queue: mp.JoinableQueue,
                                    image_queue: mp.JoinableQueue,
                                    mask_queue: mp.JoinableQueue,
                                    logger_queue: mp.JoinableQueue,
                                    transform=None):
        while True:
            try:
                image_path, mask_path = path_queue.get()
                # Thread ending condition
                if image_path is None or mask_path is None:
                    break

                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if transform is not None:
                    augmented = transform(image=image, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']

                # turning them into tensors
                image = torch.from_numpy(image).permute(2, 1, 0)
                mask = torch.from_numpy(mask).unsqueeze(0)

                image_queue.put(image)
                mask_queue.put(mask)

            except Exception as e:
                logger_queue.put(f'{str(e)} >> {image_path} && {mask_path}')
            finally:
                path_queue.task_done()

        logger_queue.put(None)

    def start(self):
        self.__populate_path_queue__()
        self.logger.start()

        for process in self.processes:
            process.start()

    def join(self):
        print("Begin Joining")

        self.path_queue.join()

        for process in self.processes:
            process.join()

        self.logger_queue.join()
        self.image_queue.join()
        self.mask_queue.join()

        print("Finished Joining")

    def __len__(self):
        return len(self.image_source)

    def __getitem__(self, index):

        while True:
            try:
                image = self.image_queue.get()
                mask = self.mask_queue.get()

                self.image_queue.task_done()
                self.mask_queue.task_done()

                return image, mask
            except queue.Empty:
                time.sleep(1) # wait a bit before trying again
                continue

