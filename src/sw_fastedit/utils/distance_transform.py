from __future__ import annotations

import logging
from typing import List, Tuple

import cupy as cp
import numpy as np
import torch

# Details here: https://docs.rapids.ai/api/cucim/nightly/api/#cucim.core.operations.morphology.distance_transform_edt
from cucim.core.operations.morphology import distance_transform_edt as distance_transform_edt_cupy
# from numpy.typing import ArrayLike
# from scipy.ndimage import distance_transform_edt

np.seterr(all="raise")

logger = logging.getLogger("sw_fastedit")

"""
CUDA enabled distance transforms using cupy
"""


# TODO replace with distance_transform_edt when MONAI 1.3.0 gets released
# def get_distance_transform(tensor: torch.Tensor, device: torch.device = None) -> torch.Tensor:
#     # The distance transform provides a metric or measure of the separation of points in the image.
#     # This function calculates the distance between each pixel that is set to off (0) and
#     # the nearest nonzero pixel for binary images
#     # http://matlab.izmiran.ru/help/toolbox/images/morph14.html
#     dimension = tensor.dim()

#     if dimension == 4:
#         tensor = tensor.squeeze(0)
#     assert len(tensor.shape) == 3 and tensor.is_cuda, "tensor.shape: {}, tensor.is_cuda: {}".format(
#         tensor.shape, tensor.is_cuda
#     )
#     with cp.cuda.Device(device.index):
#         tensor_cp = cp.asarray(tensor)
#         distance = torch.as_tensor(distance_transform_edt_cupy(tensor_cp), device=device)

#     if dimension == 4:
#         distance = distance.unsqueeze(0)
#     assert distance.dim() == dimension
#     return distance


def get_random_choice_from_tensor(
    t: torch.Tensor | cp.ndarray,
    *,
    # device: torch.device,
    max_threshold: int = None,
    size=1,
) -> Tuple[List[int], int] | None:
    device = t.device
    
    with cp.cuda.Device(device.index):
        if not isinstance(t, cp.ndarray):
            t_cp = cp.asarray(t)
        else:
            t_cp = t

        if cp.sum(t_cp) <= 0:
            # No valid distance has been found. Dont raise, just empty return
            return None, None

        # Probability transform
        if max_threshold is None:
            # divide by the maximum number of elements in a volume, otherwise we will get overflows..
            max_threshold = int(cp.floor(cp.log(cp.finfo(cp.float32).max))) / (800 * 800 * 800)

        # Clip the distance transform to avoid overflows and negative probabilities
        clipped_distance = t_cp.clip(min=0, max=max_threshold)

        flattened_t_cp = clipped_distance.flatten()

        probability = cp.exp(flattened_t_cp) - 1.0

        #print('Zrrr playing around', cp.max(probability))
        idx = cp.where(flattened_t_cp > 0)[0]
        probabilities = probability[idx] / cp.sum(probability[idx])
        assert idx.shape == probabilities.shape
        assert cp.all(cp.greater_equal(probabilities, 0))

        # Choosing an element based on the probabilities
        seed = cp.random.choice(a=idx, size=size, p=probabilities)
        dst = flattened_t_cp[seed.item()]

        # Get the elements index
        g = cp.asarray(cp.unravel_index(seed, t_cp.shape)).transpose().tolist()[0]
        index = g
        # g[0] = dst.item()
    assert len(g) == len(t_cp.shape), f"g has wrong dimensions! {len(g)} != {len(t_cp.shape)}"
    return index, dst.item()



def get_border_points_from_mask(
    mask: torch.Tensor | cp.ndarray,
    num_points: int = 1,
    max_threshold: int = None,
) -> Tuple[List[List[int]], List[float]] | None:
    device = mask.device
    
    with cp.cuda.Device(device.index):
        if not isinstance(mask, cp.ndarray):
            mask_cp = cp.asarray(mask)
        else:
            mask_cp = mask

        # Probability transform
        if max_threshold is None:
            # divide by the maximum number of elements in a volume, otherwise we will get overflows
            max_threshold = int(cp.floor(cp.log(cp.finfo(cp.float32).max))) / (800 * 800 * 800)

        # Clip the mask to avoid overflows and negative probabilities
        clipped_mask = mask_cp.clip(min=0, max=max_threshold)

        flattened_mask = clipped_mask.flatten()

        probability = cp.exp(flattened_mask) - 1.0
        idx = cp.where(flattened_mask > 0)[0]
        probabilities = probability[idx] / cp.sum(probability[idx])
        assert idx.shape == probabilities.shape
        assert cp.all(cp.greater_equal(probabilities, 0))

        # Identify border points
        border_indices = cp.where((mask_cp > 0) & (cp.roll(mask_cp, 1, axis=0) == 0))[0]
        border_probabilities = probability[border_indices] / cp.sum(probability[border_indices])

        # Ensure there are enough border points
        if len(border_indices) < num_points:
            return None

        # Choosing elements based on the probabilities
        seeds = cp.random.choice(a=border_indices, size=num_points, p=border_probabilities)
        dst_values = [flattened_mask[seed].item() for seed in seeds]

        # Get the elements' indices
        indices = [cp.asarray(cp.unravel_index(seed, mask_cp.shape)).transpose().tolist()[0] for seed in seeds]

    for g in indices:
        assert len(g) == len(mask_cp.shape), f"g has wrong dimensions! {len(g)} != {len(mask_cp.shape)}"
    
    return indices[0], dst_values[0]
