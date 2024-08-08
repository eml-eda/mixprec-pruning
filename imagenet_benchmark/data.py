# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *          Beatrice Alessandra Motetti <beatrice.motetti@polito.it>          *
# *----------------------------------------------------------------------------*

import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from .sampler import VariableBatchSamplerDDP


class RGBTransform(object):
    """Converts non-RGB images to RGB. In particular, it handles two cases:
    1) Grayscale images: it replicates the single channel three times
    2) 4D images: it discards the third channel"""
    def __call__(self, img):
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        return img


# Apply transformations
def _apply_transforms(data, transform):
    data['pixel_values'] = [transform(sample) for sample in data['image']]
    return data


# Define custom collate fn
def _custom_collate_fn(batch):
    # N.B., some values are gray-scale!
    return {
        'pixel_values': torch.stack([x['pixel_values'] if x['pixel_values'].size(0) == 3
                                     else x['pixel_values'].repeat(3, 1, 1) if x['pixel_values'].size(0) == 1
                                     else x['pixel_values'][:3]
                                     for x in batch]),
        'label': torch.tensor([x['label'] for x in batch])
    }


# Define worker init function
def _worker_init_fn(seed):
    def fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)
    return fn


def get_data(data_dir=None, source='original',
             val_split=0.1, seed=None) -> Tuple[Dataset, ...]:

    assert source in ['original', 'huggingface'], 'Unknown source'

    if data_dir is None:
        data_dir = Path.cwd() / 'data'
    else:
        data_dir = Path(data_dir)
    if not data_dir.exists():  # Check existence
        data_dir.mkdir(parents=True)

    if source == 'huggingface':
        return _get_huggingface_data(data_dir, val_split, seed)
    else:
        return _get_original_data(data_dir, val_split, seed)


def _get_huggingface_data(data_dir, val_split, seed):
    # Download data from https://huggingface.co/datasets/imagenet-1k
    # N.B., you need to login with your personal hugginface token from CL
    # using `hugginface-cli login`
    train_val_set = load_dataset('imagenet-1k', split='train',
                                 cache_dir=data_dir)
    # Validation data are used as test
    test_dataset = load_dataset('imagenet-1k', split='validation',
                                cache_dir=data_dir)

    # Define transformation and augmentations
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RGBTransform(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        RGBTransform(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Maybe split data
    if val_split > 0:
        train_val_set = train_val_set.train_test_split(val_split, seed=seed)
        train_dataset = train_val_set['train']
        val_dataset = train_val_set['test']
    else:
        train_dataset = train_val_set
        val_dataset = None

    apply_train_transform = partial(_apply_transforms, transform=train_transform)
    train_dataset.set_transform(apply_train_transform)
    apply_val_test_transform = partial(_apply_transforms, transform=test_transform)
    if val_dataset is not None:
        val_dataset.set_transform(apply_val_test_transform)
    test_dataset.set_transform(apply_val_test_transform)

    return train_dataset, val_dataset, test_dataset


def _get_original_data(data_dir, val_split, seed):
    # Define transformation and augmentations
    train_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RGBTransform(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        RGBTransform(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load data
    train_val_set = datasets.ImageFolder(root=data_dir / 'train',
                                         transform=train_transform)
    test_set = datasets.ImageFolder(root=data_dir / 'val',
                                    transform=test_transform)

    # Maybe fix seed of RNG
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    # Split train_val data in training and validation
    if val_split != 0.0:
        val_len = int(val_split * len(train_val_set))
        train_len = len(train_val_set) - val_len
        train_set, val_set = random_split(train_val_set, [train_len, val_len],
                                          generator=generator)
    else:
        train_set, val_set = train_val_set, None

    return train_set, val_set, test_set


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      source='huggingface',
                      train_batch_size=128,
                      val_batch_size=100,
                      num_workers=4,
                      seed=None,
                      sampler_fn=None,
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    assert source in ['original', 'huggingface'], 'Unknown source'
    # Unpack
    train_set, val_set, test_set = datasets

    # Maybe fix seed of RNG
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    # Maybe define worker init fn
    if seed is not None:
        worker_init_fn = _worker_init_fn(seed)
    else:
        worker_init_fn = None

    # Turn off shuffling if specific sampler is used
    shuffle = sampler_fn is None

    # With Huggingface we need to use a custom collate fn
    if source == 'huggingface':
        _collate_fn = _custom_collate_fn
    else:
        _collate_fn = None

    if sampler_fn is DistributedSampler:
        if seed is None:
            sampler = sampler_fn(train_set)  # Default seed=0, don't put None to avoid errors
        else:
            sampler = sampler_fn(train_set, seed=0)
        worker_init_fn = None
        batch_sampler = None
    elif sampler_fn is VariableBatchSamplerDDP:
        batch_sampler = sampler_fn(train_set, batch_size=train_batch_size, is_training=True)
        worker_init_fn = None
        sampler = None
        train_batch_size = 1
        shuffle = False
    elif sampler_fn is not None:
        sampler = sampler_fn
    else:
        sampler = None
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        collate_fn=_collate_fn,
        sampler=sampler,
        batch_sampler=batch_sampler,
        persistent_workers=True,
        prefetch_factor=2,
    )
    if val_set is not None:
        if sampler_fn is DistributedSampler:
            sampler = sampler_fn(val_set, seed=0)
            worker_init_fn = None
            batch_sampler = None
        elif sampler_fn is VariableBatchSamplerDDP:
            batch_sampler = sampler_fn(val_set, batch_size=val_batch_size, is_training=False)
            worker_init_fn = None
            sampler = None
            val_batch_size = 1
            shuffle = False
        elif sampler_fn is not None:
            sampler = sampler_fn
        else:
            sampler = None
        val_loader = DataLoader(
            val_set,
            batch_size=val_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
            collate_fn=_collate_fn,
            sampler=sampler,
            batch_sampler=batch_sampler,
            persistent_workers=True,
        )
    else:
        val_loader = None
    if sampler_fn is DistributedSampler:
        if seed is None:
            sampler = sampler_fn(test_set)  # Default seed=0, don't put None to avoid errors
        else:
            sampler = sampler_fn(test_set, seed=0)
        worker_init_fn = None
        batch_sampler = None
    elif sampler_fn is VariableBatchSamplerDDP:
        batch_sampler = sampler_fn(test_set, batch_size=val_batch_size, is_training=False)
        worker_init_fn = None
        sampler = None
        val_batch_size = 1
        shuffle = False
    elif sampler_fn is not None:
        sampler = sampler_fn
    else:
        sampler = None
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        collate_fn=_collate_fn,
        sampler=sampler,
        batch_sampler=batch_sampler,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader