import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import cv2
import base64

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging

logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval
from utils.cache_system import save_cached_data, load_cached_data

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from data_loader_manager.datasets import *

from torchvision.utils import make_grid, save_image

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image

from datasets import load_dataset, DownloadConfig

from cribs.crib_utils import get_crib_code


class DataLoaderConceptualCaptions(DataLoaderWrapper):
    """
    Data loader Manager for Conceptual Captions dataset
    """

    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)

    def LoadConceptualCaptions(self, module_config):
        """
        This function loads Conceptual Captions dataset
        {
          "type": "LoadConceptualCaptions", "option": "default",
          "config": {
            "conceptual_captions_path": {
                "train": "...",
                "val": "...",
            },
          },
        },
        """
        
        con_caps = load_dataset(
            "parquet",
            data_files={
                "train": module_config.config.conceptual_captions_path.train,
                "val": module_config.config.conceptual_captions_path.val,
            },
        )
        print("Q.0.1")
        print(type(con_caps))
        print(con_caps)

        self.data.conceptual_captions = EasyDict(con_caps)

    def set_dataloader(self):
        """
        This function wraps datasets into dataloader for trainers
        """

        if self.config.mode == "train":
            train_dataset_dict = {
                "data": self.data.conceptual_captions.train,
                "tokenizer": self.tokenizer,
                "mode": "train",
            }
            self.train_dataset = globals()[self.config.data_loader.dataset_type](
                self.config, train_dataset_dict
            )
             
            train_sampler = RandomSampler(
                self.train_dataset,
                num_samples=100000,
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                batch_size=self.config.train.batch_size,
                collate_fn=self.train_dataset.collate_fn,
                num_workers=4,
            )

            # COMMENT/DELETE after you finish Exercise 0
            '''
            print("Q0.4")
            #print(self.train_dataloader.keys())
            print(type(self.train_dataloader))
            train_item = next(iter(self.train_dataloader))
            print("reached here for question 0.4")
            print(train_item)
            print(train_item.keys())
            print("end printimg train_item")
            '''
            
        test_dataset_dict = {
            "data": self.data.conceptual_captions.val,
            "tokenizer": self.tokenizer,
            "mode": "test",
        }
        self.test_dataset = globals()[self.config.data_loader.dataset_type](
            self.config, test_dataset_dict
        )
        # TODO: Exercise 1.1
        # 1. Initialise test_sampler using SequentialSampler() with self.test_dataset
        # 2. Initialise self.test_dataloader using DataLoader(). 
        #   See the initialisation of self.train_dataloader elsewhere in this module.  
        #   Use 4 workers, and set the batch_size and collate_fn.
        ##### Exercise 1.1 BEGIN #####
        test_sampler = None
        self.test_dataloader = None
        exec(get_crib_code('1-1'))
        ##### Exercise 1.1 END #####
        """
        logger.info(
            "[Data Statistics]: training data loader: {};  test data loader: {}".format(
                len(self.train_dataloader), len(self.test_dataloader)
            )
            
        )
        """
