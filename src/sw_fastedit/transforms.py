# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import os
import json
import logging
import re
from typing import Dict, Hashable, List, Mapping, Tuple
from copy import deepcopy

import torch
from scipy.ndimage import gaussian_filter
from monai.config import KeysCollection
from monai.data import MetaTensor, PatchIterd
from monai.losses import DiceLoss
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    MapTransform,
    Randomizable,
    LazyTransform,
    InvertibleTransform,
    Flip,
)

from collections.abc import Hashable, Mapping, Sequence

from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utility.array import AddExtremePointsChannel
from monai.transforms.utils import extreme_points_to_image, get_extreme_points
from monai.transforms.utils_pytorch_numpy_unification import concatenate
from monai.utils.type_conversion import convert_to_dst_type
from monai.transforms.utils import check_non_lazy_pending_ops
from monai.transforms.utils_pytorch_numpy_unification import where
from monai.transforms.utility.array import SplitDim
from monai.transforms.traits import MultiSampleTrait
import numpy as np



from sw_fastedit.utils.enums import CommonKeys
from sw_fastedit.utils.enums import GanKeys

from sw_fastedit.click_definitions import LABELS_KEY, ClickGenerationStrategy
from sw_fastedit.utils.distance_transform import get_random_choice_from_tensor, get_border_points_from_mask
from monai.transforms.utils import distance_transform_edt
from sw_fastedit.utils.helper import get_global_coordinates_from_patch_coordinates, get_tensor_at_coordinates, timeit
#from FastGeodis import generalised_geodesic3d


logger = logging.getLogger("sw_fastedit")


def get_guidance_tensor_for_key_label(data, key_label, device) -> torch.Tensor:
    """Makes sure the guidance is in a tensor format."""
    tmp_gui = data.get(key_label, torch.tensor([], dtype=torch.int32, device=device))
    if isinstance(tmp_gui, list):
        tmp_gui = torch.tensor(tmp_gui, dtype=torch.int32, device=device)
    assert type(tmp_gui) is torch.Tensor or type(tmp_gui) is MetaTensor
    return tmp_gui


# TODO Franzi - one transform class - AddExtremePoints - already included in MONAI


class AddExtremePointsChanneld(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddExtremePointsChannel`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: key to label source to get the extreme points.
        background: Class index of background label, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.
        sigma: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        rescale_min: minimum value of output data.
        rescale_max: maximum value of output data.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = AddExtremePointsChannel.backend
    

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        background: int = 0,
        pert: float = 0.0,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor = 5.0,
        rescale_min: float = -1.0,
        rescale_max: float = 1.0,
        allow_missing_keys: bool = False,
        label_names = {},
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_names = label_names 
        self.background = background
        self.pert = pert
        self.points: list[tuple[int, ...]] = []
        self.label_key = label_key
        self.sigma = sigma
        self.rescale_min = rescale_min
        self.rescale_max = rescale_max
        

    def randomize(self, label: NdarrayOrTensor) -> None:
        self.points = get_extreme_points(label, rand_state=self.R, background=self.background, pert=self.pert)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        label = d[self.label_key]
        self.randomize(label[0, :])

        d['guidance'] = self.points
        return d



def get_extreme_points(
    img: NdarrayOrTensor, rand_state: np.random.RandomState | None = None, background: int = 0, pert: float = 0.0
) -> list[tuple[int, ...]]:
    """
    Generate extreme points from an image. These are used to generate initial segmentation
    for annotation models. An optional perturbation can be passed to simulate user clicks.

    Args:
        img:
            Image to generate extreme points from. Expected Shape is ``(spatial_dim1, [, spatial_dim2, ...])``.
        rand_state: `np.random.RandomState` object used to select random indices.
        background: Value to be consider as background, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.

    Returns:
        A list of extreme points, its length is equal to 2 * spatial dimension of input image.
        The output format of the coordinates is:

        indices of [1st_spatial_dim_min, 1st_spatial_dim_max, 2nd_spatial_dim_min, ..., Nth_spatial_dim_max]

    Raises:
        ValueError: When the input image does not have any foreground pixel.
    """
    check_non_lazy_pending_ops(img, name="get_extreme_points")
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    #image is label (0,1), get indices where label is not background
    indices = where(img != background)

    if np.size(indices[0]) == 0:
        raise ValueError("get_extreme_points: no foreground object in mask!")

    def _get_point(val, dim):
        """
        Select one of the indices within slice containing val.

        Args:
            val : value for comparison
            dim : dimension in which to look for value
        """
        idx = where(indices[dim] == val)[0]

        idx = idx.cpu() if isinstance(idx, torch.Tensor) else idx
        idx = rand_state.choice(idx) if rand_state is not None else idx
        pt = []
        for j in range(img.ndim):
            # add +- pert to each dimension
            val = int(indices[j][idx] + 2.0 * pert * (rand_state.rand() if rand_state is not None else 0.5 - 0.5))
            val = max(val, 0)
            val = min(val, img.shape[j] - 1)
            pt.append(val)
        return pt

    points = []
    for i in range(img.ndim):
        points.append(tuple(_get_point(indices[i].min(), i)))
        points.append(tuple(_get_point(indices[i].max(), i)))
    
 
       

    return points


class AddEmptySignalChannels(MapTransform):
    """
        Adds empty channels to the signal which will be filled with the guidance signal later.
        E.g. for two labels: 1x192x192x256 -> 3x192x192x256
    """
    def __init__(self, device, keys: KeysCollection = None):
        super().__init__(keys)
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set up the initial batch data

        in_channels = len(data[LABELS_KEY]) 
        tmp_image = data[CommonKeys.IMAGE][0:0+1, ...] #load data into temp
        assert len(tmp_image.shape) == 4
        new_shape = list(tmp_image.shape)
        new_shape[0] = in_channels
        # Set the signal to 0 for all input images
        # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
        # e.g. (3,128,128,128) for two labels
        inputs = torch.zeros(new_shape) #, device=self.device)
        inputs[0] = data[CommonKeys.IMAGE][0] #load image into inputs
        if isinstance(data[CommonKeys.IMAGE], MetaTensor):
            data[CommonKeys.IMAGE].array = inputs
        else:
            data[CommonKeys.IMAGE] = inputs

        return data
    
class AddEmptySignalChannelsLabel(MapTransform):
    """
        Adds empty channels to the signal which will be filled with the guidance signal later.
        E.g. for two labels: 1x192x192x256 -> 3x192x192x256
    """
    def __init__(self, device, keys: KeysCollection = None):
        super().__init__(keys)
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set up the initial batch data

        in_channels = len(data[LABELS_KEY]) 
        tmp_image = data[CommonKeys.LABEL][0 : 0 + 1, ...]
        assert len(tmp_image.shape) == 4
        new_shape = list(tmp_image.shape)
        new_shape[0] = in_channels
        # Set the signal to 0 for all input images
        # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
        # e.g. (3,128,128,128) for two labels
        inputs = torch.zeros(new_shape) #, device=self.device)
        inputs[0] = data[CommonKeys.LABEL][0]
        if isinstance(data[CommonKeys.LABEL], MetaTensor):
            data[CommonKeys.LABEL].array = inputs
        else:
            data[CommonKeys.LABEL] = inputs
        return data


class NormalizeLabelsInDatasetd(MapTransform):
    """
    Normalize label values according to label names dictionary

    Args:
        keys: the ``keys`` parameter will be used to get and set the actual data item to transform
        labels: all label names
        allow_missing_keys: whether to ignore it if keys are missing.
        device: device this transform shall run on

    Returns: data and also the new labels will be stored in data with key LABELS_KEY
    """
    def __init__(
        self,
        keys: KeysCollection,
        labels=None,
        allow_missing_keys: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.labels = labels
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set the labels dict in case no labels were provided
        data[LABELS_KEY] = self.labels

        for key in self.key_iterator(data):
            if "label" in key:
                try:
                    label = data[key]
                    if isinstance(label, str):
                        # Special case since label has been defined to be a string in MONAILabel
                        raise AttributeError
                except AttributeError:
                    # label does not exist - this might be a validation run
                    break

                # Dictionary containing new label numbers
                new_labels = {}
                label = torch.zeros(data[key].shape, device=self.device)
                # Making sure the range values and number of labels are the same
                for idx, (key_label, val_label) in enumerate(self.labels.items(), start=1):
                    if key_label != "background":
                        new_labels[key_label] = idx
                        label[data[key] == val_label] = idx
                    if key_label == "background":
                        new_labels["background"] = 0
                    else:
                        new_labels[key_label] = idx
                        label[data[key] == val_label] = idx

                data[LABELS_KEY] = new_labels
                if isinstance(data[key], MetaTensor):
                    data[key].array = label
                else:
                    data[key] = label
            else:
                raise UserWarning("Only the key label is allowed here!")
        return data

class AddMRIorCT():
    def __init__(
        self,
        keys: KeysCollection
    ):
        self.keys = keys
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        d = dict(data)
        #logger.info(f"image file name: {d['image_meta_dict']['filename_or_obj']}")
        #logger.info(f"label file name: {d['label_meta_dict']['filename_or_obj']}")
        numbers = re.findall(r'\d+', d['image_meta_dict']['filename_or_obj'])

        # Get the last number
        last_number = numbers[-1] if numbers else None

        print(last_number)
        if (int(last_number)>= 500):
            d[GanKeys.GLABEL] = torch.zeros((1,1))
        else:
            d[GanKeys.GLABEL] = torch.ones((1,1))
        return d


class AddGuidanceSignald(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        guidance: str = "guidance",
        sigma: int = 2,
        number_intensity_ch=3,
    ):
        super().__init__(keys, allow_missing_keys)

        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch

    def signal(self, shape, points):
        signal = np.zeros(shape, dtype=np.float32)

        flag = False
        for p in points:
            if np.any(np.asarray(p) < 0):
                continue
            if len(shape) == 3:
                signal[p[-3], p[-2], p[-1]] = 1.0
            else:
                signal[p[-2], p[-1]] = 1.0
            flag = True

        if flag:
            signal = gaussian_filter(signal, sigma=self.sigma)
            #print('min max signal',np.min(signal), np.max(signal))
            #print('signal unique', np.unique(signal))
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        return torch.Tensor(signal)[None]

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            guidance = d[self.guidance]

            guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
            if guidance and (guidance[0] or guidance[1]):
                img = img[0 : 0 + self.number_intensity_ch, ...]

                shape = img.shape[-2:] if len(img.shape) == 3 else img.shape[-3:]
                device = img.device if isinstance(img, torch.Tensor) else None
                res = self.signal(shape, guidance).to(device=device)
                #pos = self.signal(shape, guidance[0]).to(device=device)
                #neg = self.signal(shape, guidance[1]).to(device=device)
                result = torch.concat([img if isinstance(img, torch.Tensor) else torch.Tensor(img), res])
                #print('result unique', torch.unique(result))
            else:
                s = torch.zeros_like(img[0])[None]
                result = torch.concat([img, s, s])
            #result = torch.round(result)
            
            d[key] = result
        return d


class AddGuidanceSignal(MapTransform):
    """
    Add Guidance signal for input image.

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        disks: This paraemters fill spheres with a radius of sigma centered around each click.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 1,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        disks: bool = False,
        gdt: bool = False,
        spacing: Tuple = None,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.disks = disks
        self.gdt = gdt
        self.spacing = spacing
        self.device = device

    def _get_corrective_signal(self, image, guidance, key_label):
        dimensions = 3 if len(image.shape) > 3 else 2
        assert (
            type(guidance) is torch.Tensor or type(guidance) is MetaTensor
        ), f"guidance is {type(guidance)}, value {guidance}"

        if guidance.size()[0]:
            first_point_size = guidance[0].numel()
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                # Assuming the guidance has either shape (1, x, y , z) or (x, y, z)
                assert (
                    first_point_size == 4 or first_point_size == 3
                ), f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                assert first_point_size == 3, f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)

            sshape = signal.shape

            for point in guidance:
                if torch.any(point < 0):
                    continue
                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if torch.max(signal[0]) > 0:
                signal_tensor = signal[0]
                if self.sigma != 0:
                    pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                    signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                    signal_tensor = signal_tensor.squeeze(0).squeeze(0)

                signal[0] = signal_tensor
                signal[0] = (signal[0] - torch.min(signal[0])) / (torch.max(signal[0]) - torch.min(signal[0]))
                if self.disks:
                    signal[0] = (signal[0] > 0.1) * 1.0  # 0.1 with sigma=1 --> radius = 3, otherwise it is a cube

                # if self.gdt:
                #     geos = generalised_geodesic3d(image.unsqueeze(0).to(self.device),
                #                                 signal[0].unsqueeze(0).unsqueeze(0).to(self.device),
                #                                 self.spacing,
                #                                 10e10,
                #                                 1.0,
                #                                 2)




                #     signal[0] = geos[0][0]

            if not (torch.min(signal[0]).item() >= 0 and torch.max(signal[0]).item() <= 1.0):
                raise UserWarning(
                    "[WARNING] Bad signal values",
                    torch.min(signal[0]),
                    torch.max(signal[0]),
                )



            if signal is None:
                raise UserWarning("[ERROR] Signal is None")
            return signal
        else:
            if dimensions == 3:
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)
            if signal is None:
                print("[ERROR] Signal is None")
            return signal

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:

        for key in self.key_iterator(data):
            if key == "image":
                
                image = data[key]

                #image = data[key].to(torch.device("cuda:0"))
                #assert image.is_cuda 
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]

                label_key = list(data[LABELS_KEY].keys())[0]
                label_guidance = get_guidance_tensor_for_key_label(data, label_key, self.device)
                logger.debug(f"Converting guidance for label {label_key}:{label_guidance} into a guidance signal..")

                if label_guidance is not None and label_guidance.numel():
                    signal = self._get_corrective_signal(
                        image,
                        label_guidance.to(device=self.device),
                        key_label=label_key,
                    )
                    
                    assert torch.sum(signal) > 0
                else:
                    signal = self._get_corrective_signal(
                        image,
                        torch.Tensor([]).to(device=self.device),
                        key_label=label_key,
                    )

                #assert signal.is_cuda 
                #assert tmp_image.is_cuda 
                tmp_image = torch.cat([tmp_image, signal], dim=0)
                if isinstance(data[key], MetaTensor):
                    data[key].array = tmp_image
                else:
                    data[key] = tmp_image

                return data
            else:
                raise UserWarning("This transform only applies to image key")
        raise UserWarning("image key has not been been found")

class PrintShape(MapTransform):


    def __init__(
        self,
        keys: KeysCollection = None,
        allow_missing_keys: bool = False,
        prev_transform: str = None

    ):
        super().__init__(keys, allow_missing_keys)
        self.prev_transform = prev_transform

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        if self.keys != None:
            for key in self.key_iterator(data):
                print(key, self.prev_transform, data[key].shape)
        #print(data.keys())
        return data
    

class SplitDimd(MapTransform, MultiSampleTrait):
    backend = SplitDim.backend

    def __init__(
        self,
        keys: KeysCollection,
        output_postfixes: Sequence[str] | None = None,
        dim: int = 0,
        keepdim: bool = True,
        update_meta: bool = True,
        list_output: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfixes: the postfixes to construct keys to store split data.
                for example: if the key of input data is `pred` and split 2 classes, the output
                data keys will be: pred_(output_postfixes[0]), pred_(output_postfixes[1])
                if None, using the index number: `pred_0`, `pred_1`, ... `pred_N`.
            dim: which dimension of input image is the channel, default to 0.
            keepdim: if `True`, output will have singleton in the split dimension. If `False`, this
                dimension will be squeezed.
            update_meta: if `True`, copy `[key]_meta_dict` for each output and update affine to
                reflect the cropped image
            list_output: it `True`, the output will be a list of dictionaries with the same keys as original.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.output_postfixes = output_postfixes
        self.splitter = SplitDim(dim, keepdim, update_meta)
        self.list_output = list_output
        if self.list_output is None and self.output_postfixes is not None:
            raise ValueError("`output_postfixes` should not be provided when `list_output` is `True`.")

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor] | list[dict[Hashable, torch.Tensor]]:
        d = dict(data)
        all_keys = list(set(self.key_iterator(d)))
        print('all keys', all_keys)

        if self.list_output:
            output = []
            results = [self.splitter(d[key]) for key in all_keys]
            for row in zip(*results):
                new_dict = dict(zip(all_keys, row))
                # fill in the extra keys with unmodified data
                for k in set(d.keys()).difference(set(all_keys)):
                    new_dict[k] = deepcopy(d[k])
                output.append(new_dict)
            return output

        for key in all_keys:
            rets = self.splitter(d[key])
            postfixes: Sequence = list(range(len(rets))) if self.output_postfixes is None else self.output_postfixes
            if len(postfixes) != len(rets):
                raise ValueError(f"count of splits must match output_postfixes, {len(postfixes)} != {len(rets)}.")
            for i, r in enumerate(rets):
                split_key = f"{key}_{postfixes[i]}"
                if split_key in d:
                    raise RuntimeError(f"input data already contains key {split_key}.")
                d[split_key] = r
        return d






class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation
    """

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "pred":
                for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                    if key_label != "background":
                        data[f"pred_{key_label}"] = data[key][idx + 1, ...][None]
                        data[f"label_{key_label}"] = data["label"][idx + 1, ...][None]
            elif key != "pred":
                logger.info("This transform is only for pred key")
        return data


class FlipChanneld(MapTransform, InvertibleTransform, LazyTransform):
    """
    A transformation class to flip specified channels along the specified spatial axis in tensor-like data.
    This is an invertible transform and can be applied lazily.

    Args:
        keys: Keys specifying the data in the dictionary to be processed.
        spatial_axis: Spatial axis or axes along which flipping should occur.
        channels: Channels to be flipped.
        allow_missing_keys: If True, allows for keys in `keys` to be missing in the input dictionary.
        lazy: If True, the transform is applied lazily.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_axis: Sequence[int] | int | None = None,
        channels: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = Flip(spatial_axis=spatial_axis)
        self.channels = channels

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val



    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            for channel in self.channels:
                d[key][channel:channel+1] = self.flipper(d[key][channel:channel+1], lazy=lazy_)
        return d




    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            for channel in self.channels:
                d[key][channel:channel+1] = self.flipper.inverse(d[key][channel:channel+1])
        return d