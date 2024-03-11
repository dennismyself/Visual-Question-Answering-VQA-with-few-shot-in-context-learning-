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

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from data_loader_manager.module_parser import ModuleParser

class ConceptualCaptionDataset(torch.utils.data.Dataset, ModuleParser):
    """
    Base Conceptual Caption dataset class
    """

    def __init__(self, config, dataset_dict):
        logger.info(f"initialising {type(self).__name__}...")
        self.mode = dataset_dict["mode"]
        self.config = config
        self.data = dataset_dict["data"]
        self.tokenizer = dataset_dict["tokenizer"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sample = EasyDict(
            {
                "image_url": item['image_url'],
                "caption": item['caption'],
                "clip_embeddings": item['clip_embeddings'],
            }
        )
        return sample
    
    def collate_fn(self, batch):
        """
        when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
        a batch is formed as a list where each element is a defined data returned by __getitem__
        """
        # According to the settings in config file, prepare the input and output
        input_modules = self.config.model_config.input_modules.module_list
        decoder_input_modules = (
            self.config.model_config.decoder_input_modules.module_list
        )
        output_modules = self.config.model_config.output_modules.module_list

        input_data = EasyDict()
        decoder_input_data = EasyDict()
        output_data = EasyDict()

        #############################
        #  User ModuleParser to parse every sample
        #       according to what modules are selected
        #       modules are parsed in order
        #############################
        for sample in batch:
            parsed_data = self.parse_modules(
                sample, input_modules, type="input"
            )
            for key, value in parsed_data.items():
                input_data.setdefault(key, []).append(value)

            parsed_data = self.parse_modules(
                sample, decoder_input_modules, type="decoder_input"
            )
            for key, value in parsed_data.items():
                decoder_input_data.setdefault(key, []).append(value)

            parsed_data = self.parse_modules(
                sample, output_modules, type="output"
            )
            for key, value in parsed_data.items():
                output_data.setdefault(key, []).append(value)

        input_data = EasyDict(input_data)
        decoder_input_data = EasyDict(decoder_input_data)
        output_data = EasyDict(output_data)

        #############################
        #  Postprocessing Features
        #############################
        input_post_modules = (
            self.config.model_config.input_modules.postprocess_module_list
        )
        decoder_input_post_modules = (
            self.config.model_config.decoder_input_modules.postprocess_module_list
        )
        output_post_modules = (
            self.config.model_config.output_modules.postprocess_module_list
        )

        input_data = self.post_processing(input_data, input_post_modules)
        decoder_input_data = self.post_processing(
            decoder_input_data, decoder_input_post_modules
        )
        output_data = self.post_processing(output_data, output_post_modules)

        #############################
        #  Meta Features
        #############################
        image_urls = [sample.image_url for sample in batch]
        captions = [sample.caption for sample in batch]

        batched_data = EasyDict(
            {
                'image_urls': image_urls,
                'captions': captions,
            }
        )
        
        batched_data.update(input_data)
        batched_data.update(decoder_input_data)
        batched_data.update(output_data)

        return batched_data
