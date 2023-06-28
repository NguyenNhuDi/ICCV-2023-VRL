import argparse
import json
import multiprocessing as mp
import queue
import os
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
        random.shuffle(self.image_source)

        self.mask_source = glob.glob(f'{mask_dir}/*.png')
        random.shuffle(self.mask_source)

        self.transform = transform

        # Queue creations
        self.path_queue = mp.JoinableQueue()
        self.image_queue = mp.JoinableQueue()
        self.mask_queue = mp.JoinableQueue()
        self.logger_queue = mp.JoinableQueue()

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
