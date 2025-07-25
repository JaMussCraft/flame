# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import collections
import csv
import gc
import glob
import logging
import os
import pickle
import random
import re
import shutil
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenizer, model, overwrite_cache, file_path, block_size=512):

        block_size = block_size - \
            (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        directory = file_path
        cached_features_file = os.path.join(
            directory, model + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s",
                        cached_features_file)
            gc.disable()
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
                self.client_mapping = pickle.load(handle)
            gc.enable()
        else:
            logger.info("Fail to load features from cached file")

        self.data = self.examples
        self.targets = [0 for i in range(len(self.data))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(data_dir, model, overwrite_cache, block_size, tokenizer, evaluate=False):
    file_path = os.path.join(data_dir, 'test') if evaluate else os.path.join(data_dir, 'train')

    return TextDataset(tokenizer, model, overwrite_cache, file_path=file_path, block_size=block_size)


def mask_tokens(inputs, tokenizer, mlm_probability, device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone().to(device=device)
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(
        labels.shape, mlm_probability, device=device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool, device=device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.tensor(torch.bernoulli(
        probability_matrix), dtype=torch.bool).detach().to(device=device)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.tensor(torch.bernoulli(torch.full(
        labels.shape, 0.8)), dtype=torch.bool, device=device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.tensor(torch.bernoulli(torch.full(
        labels.shape, 0.5)), dtype=torch.bool, device=device) & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long).to(device=device)
    bool_indices_random = indices_random
    inputs[bool_indices_random] = random_words[bool_indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels