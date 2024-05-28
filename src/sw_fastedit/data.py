from __future__ import annotations

import glob
import logging
import os
import shutil

# from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from monai.apps import CrossValidation
from monai.data import DataLoader, Dataset, ThreadDataLoader, partition_dataset
from monai.data.dataset import PersistentDataset
from monai.data.folder_layout import FolderLayout
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    CopyItemsd,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Identityd,
    Invertd,
    Lambdad,
    LoadImaged,
    MeanEnsembled,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    SaveImaged,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SignalFillEmptyd,
    Spacingd,
    #SplitDimd,
    ToDeviced,
    ToTensord,
    VoteEnsembled,

)
from monai.utils.enums import CommonKeys

from sw_fastedit.utils.alternatingSampler import AlternatingSampler

from sw_fastedit.helper_transforms import (
    CheckTheAmountOfInformationLossByCropd,
    InitLoggerd,
    TrackTimed,
    threshold_foreground,
    cast_labels_to_zero_and_one,
)
from sw_fastedit.transforms import (
    AddEmptySignalChannels,
    AddEmptySignalChannelsLabel,
    AddExtremePointsChanneld,
    AddGuidanceSignal,
    NormalizeLabelsInDatasetd,
    SplitPredsLabeld,
    FlipChanneld,
    AddGuidanceSignald,
    AddMRIorCT,
    PrintShape,
    PrintKeys,
    SplitDimd,
)
from sw_fastedit.utils.helper import convert_mha_to_nii, convert_nii_to_mha

logger = logging.getLogger("sw_fastedit")


PET_dataset_names = ["AutoPET", "AutoPET2", "AutoPET_merged", "HECKTOR", "AutoPET2_Challenge", "AMOS"]


# def get_pre_transforms(labels: Dict, device, args, input_keys=("image", "label")):
#     """
#     Get pre-transforms for both the training and validation dataset.

#     Args:
#         labels (Dict): Dictionary containing label-related information.
#         device: Device to be used for computation.
#         args: Additional arguments.
#         input_keys (Tuple): Tuple containing input keys, default is ("image", "label").

#     Returns:
#         Tuple[Compose, Compose]: A tuple containing Compose instances for training and validation pre-transforms.
#     """
#     return Compose(get_pre_transforms_train_as_list(labels, device, args, input_keys)), Compose(
#         get_pre_transforms_val_as_list(labels, device, args, input_keys)
#     )


def get_spacing(args):
    """
    Get the voxel spacing for the specified dataset.

    Args:
        args: Additional arguments containing the dataset information.

    Returns:
        Tuple: A tuple representing the voxel spacing in (x, y, z) dimensions.

    Raises:
        UserWarning: If the specified dataset is not recognized.

    Note:
        This function returns the voxel spacing based on the dataset specified in the arguments.
        Currently supported datasets are "AutoPET," "AutoPET2," "AutoPET2_Challenge," "HECKTOR," and "MSD_Spleen."
    """
    AUTOPET_SPACING = (2.03642011, 2.03642011, 3.0)
    MSD_SPLEEN_SPACING = (2 * 0.79296899, 2 * 0.79296899, 5.0)
    HECKTOR_SPACING = (2.03642011, 2.03642011, 3.0)
    #2 options from the AMOS22 challenge paper Fabian Isensee
    #AMOS_SPACING = (1.03642011, 2.03642011, 1.0)
    #AMOS_SPACING = (3*0.69, 3*0.69, 3*2.0)
    AMOS_SPACING = (3*1.5, 3*1.0, 3*1.0) 
    # TODO Franzi Define AMOS Spacings

    if args.dataset == "AutoPET" or args.dataset == "AutoPET2" or args.dataset == "AutoPET2_Challenge":
        return AUTOPET_SPACING
    elif args.dataset == "HECKTOR":
        return HECKTOR_SPACING
    elif args.dataset == "MSD_Spleen":
        return MSD_SPLEEN_SPACING
    elif args.dataset == "AMOS":
        return AMOS_SPACING
    else:
        raise UserWarning(f"No valid dataset found: {args.dataset}")

#TODO: Franzi change transforms for AMOS dataset here
def get_pre_transforms_train_as_list(labels: Dict, device, args, input_keys, label, image):
    """
    Get a list of pre-transforms for training data.

    Args:
        labels (Dict): Dictionary of labels.
        device: The device on which to perform the transformations.
        args: Additional arguments containing information for preprocessing.
        input_keys (tuple): Tuple of input keys, default is ("image", "label").

    Returns:
        List: A list of pre-transforms for training data.

    Note:
        This function returns a list of MONAI transforms to be applied to training data.
        It includes operations such as loading images, normalization, cropping, flipping, and more.
    """
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # data Input keys have to be ["image", "label"] for train, and at least ["image"] for val
    if args.dataset in PET_dataset_names:
        t = [
            # Initial transforms on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(
                keys=input_keys,
                reader="ITKReader",
                image_only=False,
                simple_keys=True,
            ),
            #PrintShape(keys=input_keys, prev_transform='load image'),
            ToTensord(keys=input_keys, device=cpu_device, track_meta=True),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys=label, labels=labels, device=cpu_device, allow_missing_keys=True),
            #PrintShape(keys=input_keys, prev_transform='normalizeLablesin dataste'),

            Orientationd(keys=input_keys, axcodes="RAS"),
            #PrintShape(keys=input_keys, prev_transform='orientationd'),
            Spacingd(keys=image, pixdim=spacing),
            Spacingd(keys=label, pixdim=spacing, mode="nearest") if (label in input_keys) else Identityd(keys=input_keys, allow_missing_keys=True),
            #PrintShape(keys=input_keys, prev_transform='spacing'),
            CropForegroundd(
                keys=input_keys,
                source_key=image,
                select_fn=threshold_foreground,
            )
            if args.crop_foreground
            else Identityd(keys=input_keys, allow_missing_keys=True),
            #PrintShape(keys=input_keys, prev_transform='identityd'),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically
            ScaleIntensityRanged(keys=image, a_min=-45, a_max=105, b_min=0.0, b_max=1.0, clip=True)
            if args.scale_intensity_ranged
            else ScaleIntensityRangePercentilesd(
                keys=image, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0)
            if args.inferer == "SimpleInferer"
            else Identityd(keys=input_keys, allow_missing_keys=True),  # UNet needs this, 32 for 6 layers, for 7 at least 64
            #PrintShape(keys=input_keys, prev_transform='divisible pad'),
            
            #Data augmentation
            RandFlipd(keys=input_keys, spatial_axis=[0], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[1], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=input_keys, prob=0.10, max_k=3),
            #For debugging, replacing nan
            SignalFillEmptyd(input_keys),
            #PrintShape(keys=input_keys, prev_transform='signal fill empty'),
            #add ground truth extreme points to label
            AddExtremePointsChanneld(label_names = args.labels,keys = label, label_key = label,sigma = args.sigma,),
            #convert extreme points to gausian heatmap
            AddGuidanceSignald(keys=label,sigma=args.sigma),
            #PrintShape(keys=input_keys, prev_transform='add guidance signal'),
            #AddMRIorCT(keys=label),
         

            # Move to GPU
            # WARNING: Activating the line below leads to minimal gains in performance
            # However you are buying these gains with a lot of weird errors and problems
            # So my recommendation after months of fiddling is to leave this off
            # Until MONAI has fixed the underlying issues
            # ToTensord(keys=("image", "label"), device=device, track_meta=False),
        ]

        if args.debug:
            for i in range(len(t)):
                t[i] = TrackTimed(t[i])
            print(t)

    return t

#TODO: Franzi for AMOS dataset, SAME AS TRAIN BUT WITHOUT AUGMENTATION
def get_pre_transforms_val_as_list(labels: Dict, device, args, input_keys, label, image):
    """
    Get a list of pre-transforms for validation data.

    Args:
        labels (Dict): Dictionary of labels.
        device: The device on which to perform the transformations.
        args: Additional arguments containing information for preprocessing.
        input_keys (tuple): Tuple of input keys, default is ("image", "label").

    Returns:
        List: A list of pre-transforms for validation data.

    Note:
        This function returns a list of MONAI transforms to be applied to validation data.
        It includes operations such as loading images, normalization, cropping, and more.
    """
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # data Input keys have to be at least ["image"] for val
    if args.dataset in PET_dataset_names:
        t = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys=label, labels=labels, device=cpu_device, allow_missing_keys=True),
            # Only for HECKTOR, filter out the values > 1
            Lambdad(keys=label, func=cast_labels_to_zero_and_one) if (args.dataset == "HECKTOR") else Identityd(keys=input_keys, allow_missing_keys=True),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=image, pixdim=spacing),
            Spacingd(keys=label, pixdim=spacing, mode="nearest") if (label in input_keys) else Identityd(keys=input_keys, allow_missing_keys=True),
            # CheckTheAmountOfInformationLossByCropd(
            #     keys=label, roi_size=args.val_crop_size, crop_foreground=args.crop_foreground
            # )
            # if label in input_keys
            # else Identityd(keys=input_keys, allow_missing_keys=True),
            CropForegroundd(
                keys=input_keys,
                source_key=image,
                select_fn=threshold_foreground,
            )
            if args.crop_foreground
            else Identityd(keys=input_keys, allow_missing_keys=True),
            CenterSpatialCropd(keys=input_keys, roi_size=args.val_crop_size)
            if args.val_crop_size is not None
            else Identityd(keys=input_keys, allow_missing_keys=True),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically
            ScaleIntensityRanged(keys=image, a_min=-45, a_max=105, b_min=0.0, b_max=1.0, clip=True)
            if args.scale_intensity_ranged
            else ScaleIntensityRangePercentilesd(
                keys=image, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0)
            if args.inferer == "SimpleInferer"
            else Identityd(keys=input_keys, allow_missing_keys=True),
            #add ground truth extreme points to label
            AddExtremePointsChanneld(label_names = args.labels,keys = label, label_key = label,sigma = args.sigma,),
            AddGuidanceSignald(keys=label,sigma=args.sigma),
            #AddMRIorCT(keys="label")
    
        ]
        
    if args.debug:
        for i in range(len(t)):
            t[i] = TrackTimed(t[i])

    return t


def get_device(data):
    """
    Get the device information for the input data.

    Args:
        data: Input data for which the device information is retrieved.

    Returns:
        str: A string containing the device information.
    """
    return f"device - {data.device}"


#def get_click_transforms(device, args):
    """
    Get the transforms used for generating clicks during the interaction process.

    Args:
        device: Device information.
        args: Command-line arguments.

    Returns:
        Compose: A composition of transforms to be applied during the interaction process.
    """
    spacing = get_spacing(args)
    cpu_device = torch.device("cpu")
    
    logger.info(f"{device=}")
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
       #ADDS COORDINATES
        AddExtremePointsChanneld(
            label_names = args.labels,
            keys = "label", 
            label_key = "label",
            sigma = args.sigma,

             ),
        #TRANSFORMES COORDINATES INTO SPHERES
         #Overwrites the image entry
        AddGuidanceSignal(
            keys="image",
            sigma=args.sigma,
            disks=(not args.no_disks),
            gdt=args.gdt,
            spacing=spacing,
            device=device,
        ),
        FlipChanneld(keys=("image"), spatial_axis=0, channels=[1,2], allow_missing_keys=True) # Slicer clicks are flipped when saved from the monailabel server
        if args.load_from_json
        else Identityd(keys=("image"), allow_missing_keys=True),

        ToTensord(keys=("image", "label", "pred"), device=cpu_device)

        if args.sw_cpu_output
        else Identityd(keys=("pred",), allow_missing_keys=True),
    ]
    # TODO Franzi: maybe only AddGuidance and AddGuidanceSignal for the extreme points
    return Compose(t)


def get_post_transforms(labels, *, save_pred=False, output_dir=None, pretransform=None):
    """
    Get the post transforms used for processing and saving predictions.

    Args:
        labels (list): List of label names.
        save_pred (bool): Flag to indicate whether to save predictions.
        output_dir (str): Output directory for saving predictions.
        pretransform (Compose): Pre-transform to be applied before inverting the prediction.

    Returns:
        Compose: A composition of transforms for post-processing and saving predictions.
    """
    cpu_device = torch.device("cpu")
    if save_pred:
        if output_dir is None:
            raise UserWarning("output_dir may not be empty when save_pred is enabled...")
        if pretransform is None:
            logger.warning("Make sure to add a pretransform here if you want the prediction to be inverted")

    input_keys = ("pred",)
    t = [
        

        SplitDimd(keys=('image_mri'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SplitDimd(keys=('image_ct'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SplitDimd(keys=('label'))
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        CopyItemsd(keys=("pred", "pred_ep", "label_1", "label_0", "image_ct_0", "image_mri_0"), times=1,
                    names=("pred_seg_for_save", "pred_ep_for_save", "ep_for_save", "label_for_save", "image_for_save", "image_for_save"), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        #PrintKeys(),

        
        # Invertd(
        #     keys=("pred_seg_for_save", "label_for_save",),
        #     orig_keys="image",
        #     nearest_interp=False,
        #     transform=pretransform,
        # )
        # if (save_pred and pretransform is not None)
        # else Identityd(keys=input_keys, allow_missing_keys=True),
        Activationsd(keys=("pred",), softmax=True),
        AsDiscreted(
            keys="pred_seg_for_save",
            argmax=True,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        SaveImaged(
            keys=("pred_seg_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="pred_seg",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("pred_ep_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="pred_ep",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("label_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="label_seg",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("ep_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="label_ep",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("image_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="image",
            output_dtype=np.float32,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        ToTensord(keys=("image_ct", "image_mri", "label", "pred"), device=cpu_device, allow_missing_keys=True),

        
    ]
    return Compose(t)


def get_post_transforms_unsupervised(labels, device, pred_dir, pretransform):
    """
    Get the post transforms used for processing and saving unsupervised predictions without labels.

    Args:
        labels (list): List of label names. (not used but kept in function signature)
        device: Device for computation.
        pred_dir (str): Directory to save predictions.
        pretransform (Compose): Pre-transform to be applied before inverting the prediction.

    Returns:
        Compose: A composition of transforms for processing and saving unsupervised predictions.
    """
    os.makedirs(pred_dir, exist_ok=True)
    nii_layout = FolderLayout(output_dir=pred_dir, postfix="", extension=".nii.gz", makedirs=False)

    t = [
        Invertd(
            keys="pred",
            orig_keys="image",
            nearest_interp=False,
            transform=pretransform,
        ),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys="pred",
            argmax=True,
        ),
        SaveImaged(
            keys="pred",
            writer="ITKWriter",
            output_postfix="",
            output_ext=".nii.gz",
            folder_layout=nii_layout,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        ),
    ]
    return Compose(t)


def get_post_ensemble_transforms(labels, device, pred_dir, pretransform, nfolds=5, weights=None):
    """
    Get the post transforms used for processing and saving ensemble predictions.

    Args:
        labels (list): List of label names.
        device: Device for computation.
        pred_dir (str): Directory to save ensemble predictions.
        pretransform (Compose): Pre-transform to be applied before inverting the prediction.
        nfolds (int, optional): Number of folds in the ensemble. Defaults to 5.
        weights (list, optional): List of weights for ensemble voting. Defaults to None.

    Returns:
        Compose: A composition of transforms for processing and saving ensemble predictions.

    Note:
        This function returns a composition of transforms tailored for post-processing and saving ensemble predictions.
        The composition includes inversion, ensembling (mean or vote), activation, discretization, and saving.
    """
    prediction_keys = [f"pred_{i}" for i in range(nfolds)]

    os.makedirs(pred_dir, exist_ok=True)
    nii_layout = FolderLayout(output_dir=pred_dir, postfix="", extension=".nii.gz", makedirs=False)

    t = [
        Invertd(
            keys=prediction_keys,
            orig_keys="image",
            nearest_interp=False,
            transform=pretransform,
        ),
    ]

    mean_or_vote = "vote"
    if mean_or_vote == "mean":
        t += [
            EnsureTyped(keys=prediction_keys),
            MeanEnsembled(
                keys=prediction_keys,
                output_key="pred",
            ),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ]
    else:
        t += [
            EnsureTyped(keys=prediction_keys),
            Activationsd(keys=prediction_keys, softmax=True),
            AsDiscreted(keys=prediction_keys, argmax=True),
            VoteEnsembled(keys=prediction_keys, output_key="pred"),
        ]
    t += [
        SaveImaged(
            keys="pred",
            writer="ITKWriter",
            output_postfix="",
            output_ext=".nii.gz",
            folder_layout=nii_layout,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        ),
    ]
    return Compose(t)


# def get_val_post_transforms(labels, device):
#     """
#     Get post transforms for validation predictions.

#     Args:
#         labels (list): List of label names.
#         device: Device for computation.

#     Returns:
#         Compose: A composition of transforms for processing validation predictions.

#     Note:
#         The transforms include activation, discretization, and segmentation for evaluating dice score per segment/label.
#     """
#     t = [
#         Activationsd(keys="pred", softmax=True),
#         AsDiscreted(
#             keys=("pred"),
#             argmax=(True, False),
#             to_onehot=(len(labels), len(labels)),
#         ),
#         # This transform is to check dice score per segment/label
#         SplitPredsLabeld(keys="pred"),
#     ]
#     return Compose(t)





def get_AMOS_file_list(args, image_type: str) -> List[List, List, List]:
    """
    Get file lists for AutoPET dataset.

    Args:
        args: Command line arguments.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        A tuple containing lists of training, validation, and test data dictionaries.
        Each dictionary contains the paths to the image and label files.
    """
    train_images = sorted(glob.glob(os.path.join(args.input_dir,image_type, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.input_dir, image_type,"labelsTr", "*.nii.gz")))

    test_images = sorted(glob.glob(os.path.join(args.input_dir, image_type,"imagesTs", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(args.input_dir, image_type,"labelsTs", "*.nii.gz")))

    if(image_type=='CT'):
        train_data = [
            {"image_ct": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
        ]
        val_data = [{"image_ct": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
        test_data = [{"image_ct": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
    else:
        train_data = [
            {"image_mri": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
        ]
        val_data = [{"image_mri": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
        test_data = [{"image_mri": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]

    return train_data, val_data, test_data


def get_filename_without_extensions(nifti_path):
    """
    Extracts the filename without extensions from a given NIfTI file path.

    Args:
        nifti_path (str): The path to the NIfTI file.

    Returns:
        str: The filename without extensions.

    Example:
        "SUV.nii.gz" to "SUV". The resulting string represents the filename without extensions.
    """
    return Path(os.path.basename(nifti_path)).with_suffix("").with_suffix("").name








def get_data(args, image_type:str):
    """
    Retrieves data for training, validation, and testing based on the specified dataset in the command-line arguments.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.

    Returns:
        Tuple containing lists of training, validation, and test data dictionaries, each with "image" and "label" keys.
    """
    logger.info(f"{args.dataset=}")

    test_data = []

    if args.dataset == "AMOS":
        train_data, val_data, test_data = get_AMOS_file_list(args, image_type)
        



    if args.train_on_all_samples:
        train_data += val_data
        val_data = train_data
        test_data = train_data
        logger.warning("All validation data has been added to the training. Validation and Testing on them no longer makes sense.")

    logger.info(f"len(train_data): {len(train_data)}, len(val_data): {len(val_data)}, len(test_data): {len(test_data)}")

    # For debugging with small dataset size
    train_data = train_data[0 : args.limit] if args.limit else train_data
    val_data = val_data[0 : args.limit] if args.limit else val_data
    test_data = test_data[0 : args.limit] if args.limit else test_data

    return train_data, val_data, test_data


def get_test_loader(args, pre_transforms_test):
    """
    Retrieves a DataLoader for testing based on the specified command-line arguments and pre-transforms.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.
        pre_transforms_test: Pre-transforms to be applied to the test data.

    Returns:
        DataLoader for testing with a batch size of 1.
    """
    train_data, val_data, test_data = get_data(args)
    if not len(test_data):
        if len(val_data) > 0:
            test_data = val_data
        if len(train_data) > 0:
            test_data = train_data
        else:
            raise UserWarning("No valid data found..")

    total_l = len(test_data)
    test_ds = Dataset(test_data, pre_transforms_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
    )
    logger.info("{} :: Total Records used for Testing is: {}".format(args.gpu, total_l))

    return test_loader


def get_train_loader(args, pre_transforms_train_ct, pre_transforms_train_mri):
    """
    Retrieves a DataLoader for training based on the specified command-line arguments and pre-transforms.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.
        pre_transforms_train: Pre-transforms to be applied to the training data.

    Returns:
        DataLoader for training with asynchronous data loading using PersistentDataset and ThreadDataLoader.
    """
    train_data_ct, val_data_ct, test_data = get_data(args, 'CT')
    train_data_mri, val_data_mri, test_data = get_data(args, 'MRI')


    total_l_ct = len(train_data_ct) + len(val_data_ct)
    total_l_mri = len(train_data_mri) + len(val_data_mri)
    total_l = total_l_ct + total_l_mri

    train_ds_ct = PersistentDataset(train_data_ct, pre_transforms_train_ct, cache_dir=args.cache_dir)
    train_ds_mri = PersistentDataset(train_data_mri, pre_transforms_train_mri, cache_dir=args.cache_dir)
    alternatingSampler = AlternatingSampler(train_ds_ct, train_ds_mri)
    train_ds = ConcatDataset([train_ds_ct, train_ds_mri])

    train_loader = ThreadDataLoader(
        train_ds,
        shuffle=False,
        sampler = alternatingSampler,
        num_workers=args.num_workers,
        batch_size=1,
    )
    logger.info("{} :: Total Records used for Training is: {}/{}".format(args.gpu, len(train_ds), total_l))
    return train_loader


def get_val_loader(args, pre_transforms_val_ct, pre_transforms_val_mri):
    """
    Retrieves a DataLoader for validation based on the specified command-line arguments and pre-transforms.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.
        pre_transforms_val: Pre-transforms to be applied to the validation data.

    Returns:
        DataLoader for validation with asynchronous data loading using PersistentDataset and ThreadDataLoader.
    """
    train_data_ct, val_data_ct, test_data = get_data(args, 'CT')
    train_data_mri, val_data_mri, test_data = get_data(args, 'MRI')


    total_l_ct = len(train_data_ct) + len(val_data_ct)
    total_l_mri = len(train_data_mri) + len(val_data_mri)
    total_l = total_l_ct + total_l_mri


    val_ds_ct = PersistentDataset(val_data_ct, pre_transforms_val_ct, cache_dir=args.cache_dir)
    val_ds_mri = PersistentDataset(val_data_mri, pre_transforms_val_mri, cache_dir=args.cache_dir)
    val_ds = ConcatDataset([val_ds_ct, val_ds_mri])

    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=args.num_workers,
        batch_size=1,
    )
    logger.info("{} :: Total Records used for Validation is: {}/{}".format(args.gpu, len(val_ds), total_l))

    return val_loader


def get_cross_validation(args, nfolds, pre_transforms_train, pre_transforms_val):
    """
    Generates cross-validation datasets and corresponding data loaders based on the specified command-line arguments.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.
        nfolds: Number of folds for cross-validation.
        pre_transforms_train: Pre-transforms to be applied to the training data.
        pre_transforms_val: Pre-transforms to be applied to the validation data.

    Returns:
        Tuple of lists containing training and validation DataLoader instances for each fold.
    """
    folds = list(range(nfolds))

    train_data, val_data, test_data = get_data(args)

    cvdataset = CrossValidation(
        dataset_cls=PersistentDataset,
        data=train_data,
        nfolds=nfolds,
        seed=args.seed,
        transform=pre_transforms_train,
        cache_dir=args.cache_dir,
    )

    train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
    val_dss = [cvdataset.get_dataset(folds=i, transform=pre_transforms_val) for i in range(nfolds)]

    train_loaders = [
        ThreadDataLoader(
            train_dss[i],
            shuffle=True,
            num_workers=args.num_workers,
            batch_size=1,
        )
        for i in folds
    ]

    val_loaders = [
        ThreadDataLoader(
            val_dss[i],
            num_workers=args.num_workers,
            batch_size=1,
        )
        for i in folds
    ]

    return train_loaders, val_loaders


def get_metrics_loader(args, file_glob="*.nii.gz"):
    """
    Generates a metrics loader for evaluating predictions based on specified command-line arguments.

    Args:
        args: Command-line arguments specifying the labels directory, predictions directory, and other options.
        file_glob (str): File pattern to match predictions.

    Returns:
        List of dictionaries containing label and prediction file paths for evaluation.
    """
    labels_dir = args.labels_dir
    predictions_dir = args.predictions_dir
    predictions_glob = os.path.join(predictions_dir, file_glob)
    test_predictions = sorted(glob.glob(predictions_glob))
    test_datalist = []

    for pred_file_name in test_predictions:
        logger.info(f"{pred_file_name=}")
        assert os.path.exists(pred_file_name)
        file_name = get_filename_without_extensions(pred_file_name)
        label_file_name = os.path.join(labels_dir, f"{file_name}{file_glob[1:]}")
        assert os.path.exists(label_file_name)
        logger.info(f"{label_file_name=}")
        test_datalist.append({CommonKeys.LABEL: label_file_name, CommonKeys.PRED: pred_file_name})

    test_datalist = test_datalist[0 : args.limit] if args.limit else test_datalist
    total_l = len(test_datalist)
    assert total_l > 0

    logger.info("{} :: Total Records used for Dataloader is: {}".format(args.gpu, total_l))

    return test_datalist


def get_metrics_transforms(device, labels, args):
    """
    Generates a set of transforms for processing predictions during metrics calculation.

    Args:
        device: Device to which tensors are moved.
        labels: Dictionary containing labels information.
        args: Command-line arguments specifying debugging options, logging, and output directory.

    Returns:
        Composed set of transforms for metrics calculation.
    """
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    t = [
        InitLoggerd(loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir),
        LoadImaged(
            keys=["pred", "label"],
            reader="ITKReader",
            image_only=False,
        ),
        ToDeviced(keys=["pred", "label"], device=device),
        EnsureChannelFirstd(keys=["pred", "label"]),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(False, False),
            to_onehot=(len(labels), len(labels)),
        ),
    ]

    return Compose(t)
