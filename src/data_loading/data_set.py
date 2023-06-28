import argparse
import json
import logging
import multiprocessing as mp
import queue
import os
import time

import pandas as pd
import glob

import torch
import numpy as np
import torchvision
import random
from torch.utils.data import Dataset
import cv2


class WeedAndCropDataset(Dataset):
    def __init__(self, image_dir,
                 mask_dir,
                 transform=None,
                 num_procsses=1, ):
        self.image_source = glob.glob(f'{image_dir}/*.png')

        self.mask_source = glob.glob(f'{mask_dir}/*.png')

        self.transform = transform

        # Queue creations
        self.path_queue = mp.JoinableQueue()
        self.image_queue = mp.JoinableQueue()
        self.mask_queue = mp.JoinableQueue()
        self.logger_queue = mp.JoinableQueue()
        self.command_queue = mp.JoinableQueue()

    @staticmethod
    def __path_process__(command_queue: mp.JoinableQueue,
                         path_queue: mp.JoinableQueue,
                         image_source,
                         mask_source):
        while True:

            try:
                command = command_queue.get()
                if command is None:
                    break
                random.shuffle(image_source)
                random.shuffle(mask_source)

                for i in range(len(image_source)):
                    path_queue.put((image_source[i], mask_source[i]))
            except queue.Empty:
                time.sleep(1)  # sleep for a while before trying again
                continue
            else:
                command_queue.task_done()

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

    def __len__(self):
        return len(self.image_source)
