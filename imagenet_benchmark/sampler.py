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
# *----------------------------------------------------------------------------*

import copy
import math
import random
from typing import List, Iterator, Tuple, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class BaseSamplerDDP(Sampler):
    # Inspired from: https://github.com/apple/ml-cvnets/blob/main/data/sampler/base_sampler.py#L159
    """
    sharding:
    num_repeats: Repeat the training dataset samples by this factor in each epoch (aka repeated augmentation).
                 This effectively increases samples per epoch. As an example, if dataset has 10000 samples
                 and sampler.num_repeats is set to 2, then total samples in each epoch would be 20000.
                 Defaults to 1.
    trunc_rep_aug: When enabled, it restricts the sampler to load a subset of the training dataset such that
                   number of samples obtained after repetition are the same as the original dataset.
                   As an example, if dataset has 10000 samples, sampler.num_repeats is set to 2, and
                   sampler.truncated_repeat_aug_sampler is enabled, then the sampler would sample
                   10000 samples in each epoch. Defaults to False.
    """
    def __init__(self, data: Dataset, batch_size: int, is_training: bool,
                 sharding: bool = False, num_repeats: int = 1, trunc_rep_aug: bool = False,
                 disable_shuffle_sharding: bool = False) -> None:
        n_data_samples = len(data)

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        gpus_node_i = max(1, torch.cuda.device_count())

        num_samples_per_replica = int(math.ceil(n_data_samples * 1.0 / num_replicas))
        total_size = num_samples_per_replica * num_replicas

        img_indices = [idx for idx in range(n_data_samples)]
        img_indices += img_indices[: (total_size - n_data_samples)]
        assert len(img_indices) == total_size

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.shuffle = True if is_training else False
        self.epoch = 0
        self.rank = rank
        self.batch_size_gpu0 = batch_size
        self.num_replicas = num_replicas
        self.skip_sample_indices = []
        self.node_id = rank // gpus_node_i

        self.sharding = sharding
        self.num_repeats = num_repeats
        self.trunc_rep_aug = trunc_rep_aug
        self.disable_shuffle_sharding = disable_shuffle_sharding

        sample_multiplier = 1 if self.trunc_rep_aug else self.num_repeats
        self.n_samples_per_replica = num_samples_per_replica * sample_multiplier

    def get_indices_rank_i(self) -> List[int]:
        """Returns a list of indices of dataset elements for each rank to iterate over.

        ...note:
            1. If repeated augmentation is enabled, then indices will be repeated.
            2. If sharding is enabled, then each rank will process a subset of the dataset.
        """
        img_indices = copy.deepcopy(self.img_indices)
        if self.shuffle:
            random.seed(self.epoch)

            if self.sharding:
                """If we have 8 samples, say [0, 1, 2, 3, 4, 5, 6, 7], and we have two nodes,
                then node 0 will receive first 4 samples and node 1 will receive last 4 samples.

                note:
                    This strategy is useful when dataset is large and we want to process subset of dataset on each node.
                """

                # compute number pf samples per node.
                # Each node may have multiple GPUs
                # Node id = rank // num_gpus_per_rank
                samples_per_node = int(math.ceil(len(img_indices) / self.num_nodes))
                indices_node_i = img_indices[
                    self.node_id
                    * samples_per_node:(self.node_id + 1)
                    * samples_per_node
                ]

                # Ensure that each node has equal number of samples
                if len(indices_node_i) < samples_per_node:
                    indices_node_i += indices_node_i[
                        : (samples_per_node - len(indices_node_i))
                    ]

                # Note: For extremely large datasets, we may want to disable shuffling for efficient data loading
                if not self.disable_shuffle_sharding:
                    # shuffle the indices within a node.
                    random.shuffle(indices_node_i)

                if self.num_repeats > 1:
                    """Assume that we have [0, 1, 2, 3] samples in rank_i. With repeated augmentation,
                    we first repeat the samples [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] and then select 4
                    samples [0, 0, 0, 1]. Note shuffling at the beginning
                    """
                    # Apply repeated augmentation
                    n_samples_before_repeat = len(indices_node_i)
                    indices_node_i = np.repeat(indices_node_i, repeats=self.num_repeats)
                    indices_node_i = list(indices_node_i)
                    if self.trunc_rep_aug:
                        indices_node_i = indices_node_i[:n_samples_before_repeat]

                # divide the samples among each GPU in a node
                indices_rank_i = indices_node_i[
                    self.local_rank:len(indices_node_i):self.num_gpus_node_i
                ]
            else:
                """If we have 8 samples, say [0, 1, 2, 3, 4, 5, 6, 7], and we have two nodes,
                then node 0 will receive [0, 2, 4, 6] and node 1 will receive [1, 3, 4, 7].

                note:
                    This strategy is useful when each data sample is stored independently, and is
                    default in many frameworks
                """
                random.shuffle(img_indices)

                if self.num_repeats > 1:
                    # Apply repeated augmentation
                    n_samples_before_repeat = len(img_indices)
                    img_indices = np.repeat(img_indices, repeats=self.num_repeats)
                    img_indices = list(img_indices)
                    if self.trunc_rep_aug:
                        img_indices = img_indices[:n_samples_before_repeat]

                # divide the samples among each GPU in a node
                indices_rank_i = img_indices[
                    self.rank:len(img_indices): self.num_replicas
                ]
        else:
            indices_rank_i = img_indices[
                self.rank:len(self.img_indices):self.num_replicas
            ]
        return indices_rank_i

    def __len__(self):
        return (len(self.img_indices) // self.num_replicas) * (
            1 if self.trunc_rep_aug else self.num_repeats
        )

    def __iter__(self):
        raise NotImplementedError

    def set_epoch(self, epoch: int) -> None:
        """Helper function to set epoch in each sampler."""
        self.epoch = epoch

    def update_scales(
        self, epoch: int, is_master_node: bool = False, *args, **kwargs
    ) -> None:
        """Helper function to update scales in each sampler. This is typically useful in variable-batch sampler

        Subclass is expected to implement this function. By default, we do not do anything
        """
        raise NotImplementedError

    def update_indices(self, new_indices: List[int]) -> None:
        """Update indices to new indices. This function might be useful for sample-efficient training."""
        self.img_indices = new_indices


class VariableBatchSamplerDDP(BaseSamplerDDP):
    # Code inspired from: https://github.com/apple/ml-cvnets/blob/main/data/sampler/variable_batch_sampler.py#L243
    def __init__(self, dataset: Dataset, batch_size: int, is_training: bool,
                 crop_size_w: int = 224, crop_size_h: int = 224,
                 min_crop_size_w: int = 128, min_crop_size_h: int = 320,
                 max_crop_size_w: int = 128, max_crop_size_h: int = 320,
                 check_scale_div_factor: int = 32,
                 max_img_scales: int = 5,
                 scale_inc: bool = False,
                 min_scale_inc_factor: float = 1.0,
                 max_scale_inc_factor: float = 1.0,
                 scale_ep_intervals: List = [40],):
        super().__init__(dataset, batch_size, is_training)

        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h
        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w

        self.min_scale_inc_factor = min_scale_inc_factor
        self.max_scale_inc_factor = max_scale_inc_factor
        self.scale_ep_intervals = scale_ep_intervals
        self.max_img_scales = max_img_scales
        self.check_scale_div_factor = check_scale_div_factor
        self.scale_inc = scale_inc

        if is_training:
            self.img_batch_tuples = image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.num_replicas,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h,
            )
        else:
            self.img_batch_tuples = [
                (self.crop_size_h, self.crop_size_w, self.batch_size_gpu0)
            ]

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        indices_rank_i = self.get_indices_rank_i()
        start_index = 0
        n_samples_rank_i = len(indices_rank_i)
        while start_index < n_samples_rank_i:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, n_samples_rank_i)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                yield batch

    def update_scales(self, epoch: int) -> None:
        """Update the scales in variable batch sampler at specified epoch intervals during training."""
        if (epoch in self.scale_ep_intervals) and self.scale_inc:
            self.min_crop_size_w += int(
                self.min_crop_size_w * self.min_scale_inc_factor
            )
            self.max_crop_size_w += int(
                self.max_crop_size_w * self.max_scale_inc_factor
            )

            self.min_crop_size_h += int(
                self.min_crop_size_h * self.min_scale_inc_factor
            )
            self.max_crop_size_h += int(
                self.max_crop_size_h * self.max_scale_inc_factor
            )

            self.img_batch_tuples = image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.num_replicas,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h,
            )


def image_batch_pairs(
    crop_size_w: int,
    crop_size_h: int,
    batch_size_gpu0: int,
    max_scales: Optional[float] = 5,
    check_scale_div_factor: Optional[int] = 32,
    min_crop_size_w: Optional[int] = 160,
    max_crop_size_w: Optional[int] = 320,
    min_crop_size_h: Optional[int] = 160,
    max_crop_size_h: Optional[int] = 320,
    *args,
    **kwargs
) -> List[Tuple[int, int, int]]:
    """This function creates batch and image size pairs.  For a given batch size and image size, different image sizes
        are generated and batch size is adjusted so that GPU memory can be utilized efficiently.

    Args:
        crop_size_w: Base Image width (e.g., 224)
        crop_size_h: Base Image height (e.g., 224)
        batch_size_gpu0: Batch size on GPU 0 for base image
        max_scales: Number of scales.
        How many image sizes that we want to generate between min and max scale factors. Default: 5
        check_scale_div_factor: Check if image scales are divisible by this factor. Default: 32
        min_crop_size_w: Min. crop size along width. Default: 160
        max_crop_size_w: Max. crop size along width. Default: 320
        min_crop_size_h: Min. crop size along height. Default: 160
        max_crop_size_h: Max. crop size along height. Default: 320

    Returns:
        a sorted list of tuples. Each index is of the form (h, w, batch_size)

    """
    width_dims = create_intervallic_integer_list(
        crop_size_w,
        min_crop_size_w,
        max_crop_size_w,
        max_scales,
        check_scale_div_factor,
    )
    height_dims = create_intervallic_integer_list(
        crop_size_h,
        min_crop_size_h,
        max_crop_size_h,
        max_scales,
        check_scale_div_factor,
    )
    img_batch_tuples = set()
    n_elements = crop_size_w * crop_size_h * batch_size_gpu0
    for (crop_h, crop_y) in zip(height_dims, width_dims):
        # compute the batch size for sampled image resolutions with respect to the base resolution
        _bsz = max(1, int(round(n_elements / (crop_h * crop_y), 2)))

        img_batch_tuples.add((crop_h, crop_y, _bsz))

    img_batch_tuples = list(img_batch_tuples)
    return sorted(img_batch_tuples)


def create_intervallic_integer_list(
    base_val: Union[int, float],
    min_val: float,
    max_val: float,
    num_scales: Optional[int] = 5,
    scale_div_factor: Optional[int] = 1,
) -> List[int]:
    """This function creates a list of `n` integer values that scales `base_val` between
    `min_scale` and `max_scale`.

    Args:
        base_val: The base value to scale.
        min_val: The lower end of the value.
        max_val: The higher end of the value.
        n: Number of scaled values to generate.
        scale_div_factor: Check if scaled values are divisible by this factor.

    Returns:
        a sorted list of tuples. Each index is of the form (h, w, n_frames)
    """
    values = set(np.linspace(min_val, max_val, num_scales))
    values.add(base_val)
    values = [_make_divisible(v, scale_div_factor) for v in values]
    return sorted(values)


def _make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v