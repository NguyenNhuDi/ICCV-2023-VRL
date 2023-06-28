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
from torch.utils.data import Dataset


class WeedAndCropDataset(Dataset):
    def __init__(self, image_dir, instances_dir, visibility_dir, semantics_dir):
        self.image_source = glob.glob(f'{image_dir}/*.png')
        self.instance_source = glob.glob(f'{instances_dir}/*.png')
        self.visibility_source = glob.glob(f'{visibility_dir}/*.png')
        self.semantic_source = glob.glob(f'{semantics_dir}/*.png')

