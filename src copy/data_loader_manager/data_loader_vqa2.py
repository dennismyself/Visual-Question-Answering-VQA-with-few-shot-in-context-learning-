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

from cribs.crib_utils import get_crib_code


class DataLoaderVQA2(DataLoaderWrapper):
    """
    Data loader for VQA with ClipCap dataset
    """

    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)

    def LoadClipEmbeddings(self, module_config):
        """
        Load clip embeddings
        {
          "type": "LoadClipEmbeddings", "option": "default",
          "config": {
                "train": "..",
                "val": "..",
                "test": "..",
            },
        },
        """
        #############################
        #   Read Clip Embeddings
        #############################

        self.data.clip_embeddings = load_cached_data(
            self.config, "clip_embeddings"
        )
        if not self.data.clip_embeddings:
            self.data.clip_embeddings = EasyDict()
            for data_split in ["train", "val"]:
                # Read pre-extracted features
                clip_embeddings_file = module_config.config[data_split]
                logger.info(f"Reading: {clip_embeddings_file}")
                with open(clip_embeddings_file, "rb") as f:
                    self.data.clip_embeddings.update(EasyDict(pickle.load(f)))

            save_cached_data(
                self.config, self.data.clip_embeddings, "clip_embeddings"
            )

        logger.info(
            "[Data Statistics] CLIP embeddings {}".format(
                len(self.data.clip_embeddings)
            )
        )

    def LoadInContextExamples(self, module_config):
        """
        Load in-context examples for few-shot VQA
        {
          "type": "LoadInContextExamples", "option": "default",
          "config": {
                "path": "..",
            },
        },
        """
        #############################
        #   Read In-Context Examples
        #############################

        self.data.in_context_examples = EasyDict()
        # Read pre-extracted features
        in_context_examples_file = module_config.config["file_path"]
        logger.info(f"Reading: {in_context_examples_file}")
        with open(in_context_examples_file, "rb") as f:
            self.data.in_context_examples.update(EasyDict(pickle.load(f)))

        logger.info(
            "[Data Statistics] In-context examples {}".format(
                len(self.data.in_context_examples)
            )
        )


    def LoadVQA2Data(self, module_config):
        """
        Load vqa data into self.data.vqa2_data
        {
          "type": "LoadOKVQAData", "option": "default",
          "config": {
            "vqa_data_path": {
                "question_files":{
                    "train": "..",
                    "test": "..",
                },
                "annotation_files": {
                    "train": "..",
                    "test": "..",
                },
            },
            "image_data_path": {
                "train": "..",
                "valid": "..",
            },
        },
        """
        ######################
        #   Read OK-VQA data
        ######################
        def most_frequent(List):
            return max(set(List), key=List.count)

        answer_candidate_list = []
        # if self.config.mode == "test":
        #     vqa_helpers = EasyDict(
        #         {
        #             "val": VQA(
        #                 module_config.config.vqa_data_path.annotation_files.val,
        #                 module_config.config.vqa_data_path.question_files.val,
        #             ),
        #         }
        #     )
        # else:
        vqa_helpers = EasyDict(
            {
                "train": VQA(
                    module_config.config.vqa_data_path.annotation_files.train,
                    module_config.config.vqa_data_path.question_files.train,
                ),
                "val": VQA(
                    module_config.config.vqa_data_path.annotation_files.val,
                    module_config.config.vqa_data_path.question_files.val,
                ),
            }
        )

        self.data.vqa2_data = EasyDict(
            {
                "train": {},
                "val": {},
                "lookup": {},
                "vqa_helpers": vqa_helpers,
            }
        )

        for data_split, vqa_helper in vqa_helpers.items():
            vqa_helper.createIndex()
            vqa_helper.info()

            # For each data split, prepare dataset
            self.data.vqa2_data[data_split] = load_cached_data(
                self.config, "{}_data_preprocessed".format(data_split)
            )
            if not self.data.vqa2_data[data_split]:
                # This split data is not cached
                self.data.vqa2_data[data_split] = EasyDict({})  # re-initialise
                # Create list of images from helper
                img_data_path = module_config.config.image_data_path[
                    data_split
                ]
                img_list = []
                for imgId in vqa_helper.imgToQA.keys():
                    dataSubType = vqa_helper.dataSubType
                    imgFilename = (
                        "COCO_"
                        + dataSubType
                        + "_"
                        + str(imgId).zfill(12)
                        + ".jpg"
                    )
                    img_path = os.path.join(img_data_path, imgFilename)
                    img_list.append((imgId, img_path))
                    if self.config.data_loader.dummy_dataloader:
                        # Load only a few samples for testing
                        if len(img_list) > 20:
                            break

                # Create entries for each question and related answers
                self.data.vqa2_data[data_split].data_items = []
                for imgId, img_path in tqdm(img_list):
                    # avoid error in splitting: must remove ".." in "../path/to/file"
                    # img_key = img_p.replace('..', '').split('.')[0].split('_')[-1]
                    img_key = imgId

                    img_key_full = str(img_key).zfill(12)
                    # img = cv2.imread(img_path)
                    img = []

                    related_question_ids = vqa_helper.getQuesIds(
                        imgIds=[imgId]
                    )
                    related_answers = vqa_helper.loadQA(
                        ids=related_question_ids
                    )
                    related_question_and_answers = vqa_helper.returnQA(
                        related_answers
                    )

                    for question_and_answer in related_question_and_answers:

                        # TODO: Exercise 2.1 Prepare VQA entry data from `question_and_answer` object
                        # Create a variable `entry_data` of type EasyDict() that has the following keys:
                        # question: the question
                        # question_id: question identifier in VQA 2.0
                        # answers: list of annotated answers ; make sure not to include answers that are empty strings
                        # gold_answer: the most frequent answer in the answer list. You may use the `most_frequent()` function defined above to select the gold answer
                        # img_path: `img_path` as defined above
                        # img_key_full: `img_key_full` as defined above
                        # img_key: `img_key` as defined above
                        # img: `img` as defined above

                        # For each question and related answers, create an entry
                        
                        entry_data = EasyDict()

                        ##### Exercise 2.1 BEGIN #####
                        self.crib_dict = {}
                        exec(get_crib_code("2-1"))
                        entry_data = self.crib_dict.get('entry_data', entry_data)
                       
                        ##### Exercise 2.1 END #####

                        self.data.vqa2_data[data_split].data_items.append(
                            entry_data
                        )

                        # Collect answer candidates for evaluation
                        for ans in list(
                            question_and_answer["answers"].values()
                        ):
                            if ans not in answer_candidate_list:
                                answer_candidate_list.append(ans)
                                # if data_split == 'test':
                                #     print(ans, 'is added from test set!')

                # After building the data split, save to cache
                save_cached_data(
                    self.config,
                    self.data.vqa2_data[data_split],
                    "{}_data_preprocessed".format(data_split),
                )

            for entry_data in self.data.vqa2_data[data_split].data_items:
                self.data.vqa2_data["lookup"][
                    str(entry_data.question_id)
                ] = entry_data

            # Report statistics
            logger.info(
                "[Data statistics] split: {}  entries: {}".format(
                    data_split, len(self.data.vqa2_data[data_split].data_items)
                )
            )

        # Save answer candidate list
        self.data.vqa2_data.answer_candidate_list = answer_candidate_list

        self.data.vqa_data = self.data.vqa2_data

    def set_dataloader(self):
        """
        This function wraps datasets into dataloader for trainers
        """
        if self.config.mode == "train":
            train_dataset_dict = {
                "data": self.data.vqa_data.train,
                "vinvl_features": self.data.get("vinvl_features", None),
                "ocr_features": self.data.get("ocr_features", None),
                "clip_embeddings": self.data.get("clip_embeddings", None),
                "answer_candidate_list": self.data.vqa_data.answer_candidate_list,
                "tokenizer": self.tokenizer,
                "decoder_tokenizer": self.decoder_tokenizer,
                "feature_extractor": self.feature_extractor,
                "image_preprocessor": self.image_preprocessor,
                "mode": "train",
            }
            self.train_dataset = globals()[self.config.data_loader.dataset_type](
                self.config, train_dataset_dict
            )
            train_sampler = RandomSampler(self.train_dataset)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                batch_size=self.config.train.batch_size,
                collate_fn=self.train_dataset.collate_fn,
                num_workers=8,
            )
            logger.info(
            "[Data Statistics]: training data loader: {}".format(
                len(self.train_dataloader)
            )
        )
        # for i in self.train_dataloader:
        #     print(i)
        #     input()

        test_dataset_dict = {
            "data": self.data.vqa_data.val,
            "vinvl_features": self.data.get("vinvl_features", None),
            "ocr_features": self.data.get("ocr_features", None),
            "clip_embeddings": self.data.get("clip_embeddings", None),
            "in_context_examples": self.data.get("in_context_examples", None),
            "answer_candidate_list": self.data.vqa_data.answer_candidate_list,
            "tokenizer": self.tokenizer,
            "decoder_tokenizer": self.decoder_tokenizer,
            "feature_extractor": self.feature_extractor,
            "image_preprocessor": self.image_preprocessor,
            "mode": "test",
        }
        self.test_dataset = globals()[self.config.data_loader.dataset_type](
            self.config, test_dataset_dict
        )
        

        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.config.valid.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=0,
        )
        # for i in self.test_dataloader:
        #     pprint(i)
        #     input()
        
        logger.info(
            "[Data Statistics]: test data loader: {}".format(
                len(self.test_dataloader)
            )
        )
