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

# Code extension and modification by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology #
# zdravko.marinov@kit.edu #
# Further code extension and modification by B.Sc. Matthias Hadlich, Karlsuhe Institute of Techonology #
# matthiashadlich@posteo.de #

from __future__ import annotations

import logging
import os
import random
from collections import OrderedDict
from functools import reduce
from pickle import dump
from typing import Iterable, List
import sys

import cupy as cp
import numpy as np
import torch
from ignite.engine import Events
from ignite.handlers import TerminateOnNan
from monai.data import set_track_meta
from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointLoader,
    CheckpointSaver,
    GarbageCollector,
    IgniteMetricHandler,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import MSEMetric
from monai.metrics import SurfaceDiceMetric
from monai.networks.nets.dynunet import DynUNet
from monai.optimizers.novograd import Novograd
from monai.transforms import Compose
from monai.utils import set_determinism

from sw_fastedit.data import (
    get_click_transforms,
    get_post_transforms,
    get_pre_transforms_train_as_list,
    get_pre_transforms_val_as_list,
    get_train_loader,
    get_val_loader,
    get_test_loader,
)
from sw_fastedit.interaction import Interaction
from sw_fastedit.utils.helper import count_parameters, is_docker, run_once, handle_exception

logger = logging.getLogger("sw_fastedit")
output_dir = None


def get_optimizer(optimizer: str, lr: float, network):
    """
    Get an optimizer for the given neural network.

    Parameters:
        optimizer (str): The optimizer to use. Options: "Novograd" or "Adam" (default).
        lr (float): The learning rate for the optimizer.
        network: The neural network for which the optimizer is created.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer with the provided learning rate.

    Example:
        # Get an Adam optimizer with a learning rate of 0.001 for a neural network
        optimizer = get_optimizer("Adam", 0.001, my_neural_network)
    """
    if optimizer == "Novograd":
        optimizer = Novograd(network.parameters(), lr)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), lr)
    return optimizer


def get_loss_function(loss_args, loss_kwargs=None):
    """
    Get a loss function for a semantic segmentation task.

    Parameters:
        loss_args (str): The type of loss function. Options: "DiceCELoss" or "DiceLoss".
        loss_kwargs (dict, optional): Additional keyword arguments for the loss function.
        Default loss_kwargs: {squared_pred: True, include_background:True}
        -   squared_pred enables faster convergence, possibly even better results in the long run

    Returns:
        torch.nn.modules.loss: A callable instance of the specified loss function.

    Example:
        # Get a DiceCELoss with default arguments
        loss_fn = get_loss_function("DiceCELoss")

        # Get a DiceLoss with custom arguments
        custom_loss_kwargs = {"squared_pred": False, "include_background": False}
        loss_fn_custom = get_loss_function("DiceLoss", loss_kwargs=custom_loss_kwargs)

    TODO Franzi:
        # Define the Losses from the paper (L_ext, L_seg, L_d, L_adv)
    """
    if loss_kwargs is None:
        loss_kwargs = {}
    if loss_args == "DiceCELoss":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, **loss_kwargs)
    elif loss_args == "DiceLoss":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, **loss_kwargs)
    return loss_function

def get_loss_function_extreme_points(loss_ep_args, loss_kwargs=None):
    """
    Get loss function for the extreme points (default: 6)

    """
    if loss_kwargs is None:
        loss_kwargs = {}
    if loss_ep_args == "mean_squared_error":
        loss_function = MSEMetric(to_onehot_y=True, softmax=True, **loss_kwargs)
    return loss_function


def get_network(network_str: str, labels: Iterable, non_interactive: bool = False):

    """
    Get a network for semantic segmentation.

    Parameters:
        network_str (str): The type of network. Options: "dynunet", "smalldynunet", "bigdynunet", "hugedynunet".
        labels (Iterable): List of label names.
        non_interactive (bool, optional): Flag indicating whether the network is used in non-interactive mode.

    Returns:
        nn.Module: An instance of the specified U-Net-based neural network.

    Additional Information:
        in_channels (int): The input channels for the network. For interactive runs, it is 1 + len(labels),
            where each additional channel represents a guidance signal per label with the size of the image.
            For non-interactive runs, it is 1, representing the image.
        out_channels (int): The number of output channels, equal to len(labels).

    Example:
        # Get a DynUNet with default configuration
        my_labels = ["tumor", "background"]
        unet_model = get_network("dynunet", my_labels)

    TODO Franzi:
        # Define the model architecture as in the paper
    """


    in_channels = 1 if non_interactive else 1 + len(labels)
    out_channels = len(labels)

    if network_str == "dynunet":
        network = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    elif network_str == "smalldynunet":
        network = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3],
            strides=[1, 2, [2, 2, 1]],
            upsample_kernel_size=[2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    elif network_str == "bigdynunet":
        network = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

    logger.info(f"Selected network {network.__class__.__qualname__}")
    logger.info(f"Number of parameters: {count_parameters(network):,}")

    return network


def get_inferers(
    inferer: str,
    *,
    sw_roi_size,
    train_crop_size,
    val_crop_size,
    train_sw_batch_size,
    val_sw_batch_size,
    train_sw_overlap=0.25,
    val_sw_overlap=0.25,
    cache_roi_weight_map: bool = True,
    device="cpu",
    sw_cpu_output=False,
):
    """
    Retrieves training and evaluation inferers based on the specified inference strategy.

    Args:
        inferer (str): The type of inferer, either "SimpleInferer" or "SlidingWindowInferer".
        sw_roi_size: Region of interest size for the sliding window strategy.
        train_crop_size: Crop size for training data.
        val_crop_size: Crop size for validation data.
        train_sw_batch_size: Batch size for training with sliding window.
        val_sw_batch_size: Batch size for validation with sliding window.
        train_sw_overlap (float, optional): SW-overlap ratio for training with sliding window (default is 0.25).
        val_sw_overlap (float, optional): SW-overlap ratio for validation with sliding window (default is 0.25).
        cache_roi_weight_map (bool, optional): Whether to pre-compute the ROI weight map. (default is True).
        device (str, optional): Device for computation, e.g., "cpu" or "cuda" (default is "cpu").
        sw_cpu_output (bool, optional): Enable sliding window output on the CPU (default is False).

    Returns:
        Tuple[Inferer, Inferer]: Training and evaluation inferers based on the specified strategy.
    """
    if inferer == "SimpleInferer":
        train_inferer = SimpleInferer()
        eval_inferer = SimpleInferer()
    elif inferer == "SlidingWindowInferer":
        # train_batch_size is limited due to this bug: https://github.com/Project-MONAI/MONAI/issues/6628
        assert train_crop_size is not None
        train_batch_size = max(
            1,
            min(
                reduce(
                    lambda x, y: x * y,
                    [round(train_crop_size[i] / sw_roi_size[i]) for i in range(len(sw_roi_size))],
                ),
                train_sw_batch_size,
            ),
        )
        logger.info(f"{train_batch_size=}")
        average_sample_shape = (300, 300, 400) # AutoPET specific!
        if val_crop_size is not None:
            average_sample_shape = val_crop_size

        val_batch_size = max(
            1,
            min(
                reduce(
                    lambda x, y: x * y,
                    [round(average_sample_shape[i] / sw_roi_size[i]) for i in range(len(sw_roi_size))],
                ),
                val_sw_batch_size,
            ),
        )
        logger.info(f"{val_batch_size=}")

        sw_params = {
            "roi_size": sw_roi_size,
            "mode": "gaussian",
            "cache_roi_weight_map": cache_roi_weight_map,
        }

        if sw_cpu_output:
            logger.warning("Enabling Sliding Window output on the CPU")
            logger.warning(
                "Note that this only works well for validation! For training AMP has to be turned off and it has no real effect"
            )
            sw_params.update({"sw_device": device, "device": "cpu"})
        train_inferer = SlidingWindowInferer(sw_batch_size=train_batch_size, overlap=train_sw_overlap, **sw_params)
        eval_inferer = SlidingWindowInferer(sw_batch_size=val_batch_size, overlap=val_sw_overlap, **sw_params)
    return train_inferer, eval_inferer


def get_scheduler(optimizer, scheduler_str: str, epochs_to_run: int):
    """
    Retrieves a learning rate scheduler based on the specified scheduler strategy.

    Args:
        optimizer: The optimizer for which the scheduler will be used.
        scheduler_str (str): The type of scheduler, one of "MultiStepLR", "PolynomialLR", or "CosineAnnealingLR".
        epochs_to_run (int): The total number of epochs to run during training.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: A learning rate scheduler instance based on the chosen strategy.

    Raises:
        ValueError: If an unsupported scheduler type is provided.
    """

    if scheduler_str == "MultiStepLR":
        steps = 4 # AutoPET specific!
        steps_per_epoch = round(epochs_to_run / steps)
        if steps_per_epoch < 1:
            logger.error(f"Chosen number of epochs {epochs_to_run}/{steps} < 0")
            milestones = range(0, epochs_to_run)
        else:
            milestones = [num for num in range(0, epochs_to_run) if num % round(steps_per_epoch) == 0][1:]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.333)
    elif scheduler_str == "PolynomialLR":
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs_to_run, power=2)
    elif scheduler_str == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_run, eta_min=1e-8)
    return lr_scheduler


def get_val_handlers(sw_roi_size: List, inferer: str, gpu_size: str, *,garbage_collector=True, non_interactive=False):
    """
    Retrieves a list of event handlers for validation in a MONAI training workflow.

    Args:
        sw_roi_size (List): The region of interest size for the sliding window strategy.
        inferer (str): The type of inferer, e.g., "SimpleInferer" or "SlidingWindowInferer".
        gpu_size (str): The GPU size, one of "large" or any other value, e.g., "small".
        garbage_collector (bool, optional): Whether to include the GarbageCollector event handler (default is True).
        non_interactive (bool, optional): Whether the training loop is non-interactive, e.g., without clicks (default is False).

    Returns:
        List[Event_Handler]: A list of event handlers for validation in a MONAI training workflow.

    Notes:
        - The returned list includes a StatsHandler and an optional GarbageCollector event handler.
        - The GarbageCollector event handler is triggered based on the provided parameters.

    Raises:
        ValueError: If an unsupported inferer type is provided.

    References:
        [1] https://github.com/Project-MONAI/MONAI/issues/3423

    TODO Franzi:
        # Set the iterations = 1 and it is done
    """
    every_x_iterations = 2
    if non_interactive:
        every_x_iterations *= 10

    if sw_roi_size[0] < 128:
        val_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED
    else:
        val_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED

    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if garbage_collector:
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event=val_trigger_event)
        val_handlers.append(iteration_gc)

    return val_handlers


def get_train_handlers(
    lr_scheduler,
    evaluator,
    val_freq,
    eval_only: bool,
    sw_roi_size: List,
    inferer: str,
    gpu_size: str,
    garbage_collector=True,
    non_interactive=False,
):
    """
    Retrieves a list of event handlers for training in a MONAI training workflow.

    Args:
        lr_scheduler: The learning rate scheduler for the optimizer.
        evaluator: The evaluator for validation during training.
        val_freq: The frequency of validation in terms of iterations or epochs.
        eval_only (bool): Flag indicating if training is for evaluation only.
        sw_roi_size (List): The region of interest size for the sliding window strategy.
        inferer (str): The type of inferer, e.g., "SimpleInferer" or "SlidingWindowInferer".
        gpu_size (str): The GPU size, one of "large" or any other value.
        garbage_collector (bool, optional): Whether to include the GarbageCollector event handler (default is True).
        non_interactive (bool, optional): Whether the environment is non-interactive (default is False).

    Returns:
        List[Event_Handler]: A list of event handlers for training in a MONAI training workflow.

    Notes:
        - The returned list includes an LrScheduleHandler, ValidationHandler, StatsHandler, and an optional GarbageCollector event handler.
        - The GarbageCollector event handler is triggered based on the provided parameters.

    Raises:
        ValueError: If an unsupported inferer type is provided.

    References:
        [1] https://github.com/Project-MONAI/MONAI/issues/3423

    TODO Franzi:
        # Set the iterations = 1 and it is done
    """
    every_x_iterations = 4
    if non_interactive:
        every_x_iterations *= 10

    if sw_roi_size[0] <= 128:
        train_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED
    else:
        train_trigger_event = (
            Events.ITERATION_COMPLETED(every=every_x_iterations*2) if gpu_size == "large" else Events.ITERATION_COMPLETED(every=every_x_iterations)
        )

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(
            validator=evaluator,
            interval=val_freq,
            epoch_level=(not eval_only),
        ),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if garbage_collector:
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event=train_trigger_event)
        train_handlers.append(iteration_gc)

    return train_handlers


def get_key_metric(str_to_prepend="") -> OrderedDict:
    """
    Retrieves key metrics, particularly Mean Dice, for use in a MONAI training workflow.

    Args:
        str_to_prepend (str, optional): A string to prepend to the metric name (default is an empty string).

    Returns:
        OrderedDict: An ordered dictionary containing key metrics for training and evaluation.

    TODO Franzi:
        # Add MXA
    """
    key_metrics = OrderedDict()
    key_metrics[f"{str_to_prepend}dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False, save_details=False
    )
    return key_metrics


def get_additional_metrics(labels, include_background=False, loss_kwargs=None, str_to_prepend=""):
    """
    Retrieves additional metrics, including DiceCELoss and SurfaceDiceMetric, for use in a MONAI training workflow.

    Args:
        labels: A list of class labels for the segmentation task.
        include_background (bool, optional): Whether to include a background class in the metric computation (default is False).
        loss_kwargs (dict, optional): Additional keyword arguments for configuring the loss function (default is None).
        str_to_prepend (str, optional): A string to prepend to the metric names (default is an empty string).

    Returns:
        OrderedDict: An ordered dictionary containing additional metrics for evaluation.

    Notes:
        - The returned dictionary includes metrics for DiceCELoss and SurfaceDiceMetric.
        - The metric names can be customized by providing a string to prepend.
        - Loss function and SurfaceDiceMetric are configured with specific parameters.

    TODO Franzi:
        # Or add MXA here
    """
    if loss_kwargs is None:
        loss_kwargs = {}
    mid = "with_bg_" if include_background else "without_bg_"
    loss_function = DiceCELoss(softmax=True, **loss_kwargs)

    loss_function_metric_ignite = IgniteMetricHandler(
        loss_fn=loss_function,
        output_transform=from_engine(["pred", "label"]),
        save_details=False,
    )
    amount_of_classes = len(labels) if include_background else (len(labels) - 1)
    class_thresholds = (0.5,) * amount_of_classes
    surface_dice_metric = SurfaceDiceMetric(
        include_background=include_background,
        class_thresholds=class_thresholds,
        reduction="mean",
        get_not_nans=False,
        use_subvoxels=True,
    )
    surface_dice_metric_ignite = IgniteMetricHandler(
        metric_fn=surface_dice_metric,
        output_transform=from_engine(["pred", "label"]),
        save_details=False,
    )

    additional_metrics = OrderedDict()
    additional_metrics[f"{str_to_prepend}{loss_function.__class__.__name__.lower()}"] = loss_function_metric_ignite
    additional_metrics[f"{str_to_prepend}{mid}surface_dice"] = surface_dice_metric_ignite

    return additional_metrics


def get_test_evaluator(
    args,
    network,
    inferer,
    device,
    val_loader,
    post_transform,
    resume_from="None",
) -> SupervisedEvaluator:
    """
    Retrieves a supervised evaluator for testing in a MONAI training workflow.

    Args:
        args: Command-line arguments and configuration settings.
        network: The model to be evaluated.
        inferer: The inference strategy or inferer.
        device: The computing device (e.g., "cuda" or "cpu") on which to run the evaluation.
        val_loader: The data loader for the validation/testing dataset.
        post_transform: Post-processing transformation for the output predictions.
        resume_from (str, optional): Path to a checkpoint file to resume training from (default is "None").

    Returns:
        SupervisedEvaluator: An instance of the supervised evaluator for testing.

    TODO Franzi:
        # SupervisedEvaluator and Trainer originate from PyTorch Ignite
    """
    init(args)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        val_handlers=get_val_handlers(
            sw_roi_size=args.sw_roi_size,
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=(not args.non_interactive),
        ),
    )

    save_dict = {
        "net": network,
    }

    if resume_from != "None":
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        logger.info(f"CWD: {os.getcwd()}")
        map_location = device
        checkpoint = torch.load(resume_from)
        logger.info(f"{checkpoint.keys()=}")
        network.load_state_dict(checkpoint["net"])
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        handler(evaluator)

    return evaluator


def get_supervised_evaluator(
    args,
    network,
    inferer,
    device,
    val_loader,
    loss_function,
    click_transforms,
    post_transform,
    key_val_metric,
    additional_metrics,
) -> SupervisedEvaluator:
    """
    Retrieves a supervised evaluator for validation in a MONAI training workflow.

    Args:
        args: Command-line arguments and configuration settings.
        network: The model to be evaluated.
        inferer: The inference strategy or inferer.
        device: The computing device (e.g., "cuda" or "cpu") on which to run the evaluation.
        val_loader: The data loader for the validation dataset.
        loss_function: The loss function for evaluation.
        click_transforms: Data transforms for interactions (e.g., clicks).
        post_transform: Post-processing transformation for the output predictions.
        key_val_metric: The key validation metric.
        additional_metrics: Additional metrics for evaluation.

    Returns:
        SupervisedEvaluator: An instance of the supervised evaluator for validation.

    TODO Franzi:
        # max_interactions = 1 (all extreme points)
    """

    init(args)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_val,
            transforms=click_transforms,
            train=False,
            label_names=args.labels,
            max_interactions=args.max_val_interactions,
            save_nifti=args.save_nifti,
            nifti_dir=args.data_dir,
            loss_function=loss_function,
            nifti_post_transform=post_transform,
            click_generation_strategy=args.val_click_generation,
            stopping_criterion=args.val_click_generation_stopping_criterion,
            non_interactive=args.non_interactive,
            # add extreme points argument TODO Franzi
        )
        if not args.non_interactive
        else None,
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=key_val_metric,
        additional_metrics=additional_metrics,
        val_handlers=get_val_handlers(
            sw_roi_size=args.sw_roi_size,
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )
    return evaluator



def get_ensemble_evaluator(
    args, networks, inferer, device, val_loader, post_transform, resume_from="None", nfolds=5
) -> EnsembleEvaluator:
    """
    Retrieves an ensemble evaluator for validation in a MONAI training workflow.

    Args:
        args: Command-line arguments and configuration settings.
        networks: List of neural network models to be evaluated in the ensemble.
        inferer: The inference strategy or inferer for the ensemble.
        device: The computing device (e.g., "cuda" or "cpu") on which to run the evaluation.
        val_loader: The data loader for the validation dataset.
        post_transform: Post-processing transformation for the ensemble predictions.
        resume_from (str, optional): Path to a directory containing individual checkpoint files for each network (default is "None").
        nfolds (int, optional): Number of folds or networks in the ensemble (default is 5).

    Returns:
        EnsembleEvaluator: An instance of the ensemble evaluator for validation.
    """

    init(args)

    device = torch.device(f"cuda:{args.gpu}")
    prediction_keys = [f"pred_{i}" for i in range(nfolds)]

    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=val_loader,
        networks=networks,
        inferer=inferer,
        postprocessing=post_transform,
        pred_keys=prediction_keys,
        amp=args.amp,
    )

    if resume_from != "None":
        logger.info(f"{args.gpu}:: Loading Networks...")
        logger.info(f"CWD: {os.getcwd()}")
        resume_path = os.path.abspath(resume_from)
        logger.info(f"{resume_path=}")

        for i in range(nfolds):
            file_path = os.path.join(resume_path, f"{i}.pt")
            logger.info(f"{file_path=}")
            checkpoint = torch.load(file_path)
            networks[i].load_state_dict(checkpoint["net"])
    return evaluator


def get_trainer(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")

    pre_transforms_val = Compose(get_pre_transforms_val_as_list(args.labels, device, args))
    if args.use_test_data_for_validation:
        val_loader = get_test_loader(args, pre_transforms_val)
    else:
        val_loader = get_val_loader(args, pre_transforms_val)

    click_transforms = get_click_transforms(sw_device, args) # TODO Franzi - extreme clicks generation
    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    network = get_network(args.network, args.labels, args.non_interactive).to(sw_device)
    train_inferer, eval_inferer = get_inferers(
        args.inferer,
        sw_roi_size=args.sw_roi_size,
        train_crop_size=args.train_crop_size,
        val_crop_size=args.val_crop_size,
        train_sw_batch_size=args.train_sw_batch_size,
        val_sw_batch_size=args.val_sw_batch_size,
        train_sw_overlap=args.train_sw_overlap,
        val_sw_overlap=args.val_sw_overlap,
    )

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_function = get_loss_function(loss_args=args.loss, loss_kwargs=loss_kwargs)
    optimizer = get_optimizer(args.optimizer, args.learning_rate, network)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)

    val_key_metric = get_key_metric(str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )

    evaluator = get_supervised_evaluator(
        args,
        network=network,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader,
        loss_function=loss_function,
        click_transforms=click_transforms,
        post_transform=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
    )

    pre_transforms_train = Compose(get_pre_transforms_train_as_list(args.labels, device, args))
    train_loader = get_train_loader(args, pre_transforms_train)
    train_key_metric = get_key_metric(str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers(
        lr_scheduler,
        evaluator,
        args.val_freq,
        args.eval_only,
        args.sw_roi_size,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_train,
            transforms=click_transforms,
            train=True,
            label_names=args.labels,
            max_interactions=args.max_train_interactions,
            save_nifti=args.save_nifti,
            nifti_dir=args.data_dir,
            loss_function=loss_function,
            nifti_post_transform=post_transform,
            click_generation_strategy=args.train_click_generation,
            stopping_criterion=args.train_click_generation_stopping_criterion,
            iteration_probability=args.train_iteration_probability,
            loss_stopping_threshold=args.train_loss_stopping_threshold,
            non_interactive=args.non_interactive,
        )
        if not args.non_interactive
        else None,
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,
    )

    if not args.eval_only:
            save_dict = {
                "trainer": trainer,
                "net": network,
                "opt": optimizer,
                "lr": lr_scheduler,
            }
    else:
        save_dict = {
            "net": network,
        }

    if ensemble_mode:
        save_dict = {
            "net": network,
        }

    if not ensemble_mode:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_interval=args.save_interval,
            save_final=True,
            final_filename="checkpoint.pt",
            save_key_metric=True,
            n_saved=2,
            file_prefix="train",
        ).attach(trainer)
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="pretrained_deepedit_" + args.network + "-final.pt",
        ).attach(evaluator)
    else:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            file_prefix=file_prefix,
        ).attach(evaluator)

    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            del save_dict["opt"]
            del save_dict["lr"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr

    return trainer, evaluator, train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics


@run_once
def init(args):
    """
    Initializes the environment with configuration settings and global variables.

    Args:
        args: Command-line arguments and configuration settings.

    Returns:
        None

    Notes:
        - This function is intended to be executed only once during the program's lifetime.

    """

    global output_dir
    # for OOM debugging
    output_dir = args.output_dir
    sys.excepthook = handle_exception

    if not is_docker():
        torch.set_num_threads(int(os.cpu_count() / 3))  # Limit number of threads to 1/3 of resources
    if args.limit_gpu_memory_to != -1:
        limit = args.limit_gpu_memory_to
        assert limit > 0 and limit < 1, f"Percentage GPU memory limit is invalid! {limit} > 0 or < 1"
        torch.cuda.set_per_process_memory_fraction(limit, args.gpu)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    # DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING..
    # I WARNED YOU..
    set_track_meta(True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    set_determinism(seed=args.seed)

    if not is_docker():
        with cp.cuda.Device(args.gpu):
            cp.random.seed(seed=args.seed)

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        np.seterr(all="raise")


def oom_observer(device, alloc, device_alloc, device_free):
    """
    Observes and handles out-of-memory (OOM) events on a specified CUDA device.

    Args:
        device (torch.device): The CUDA device where the OOM event occurred.
        alloc (int): Allocated memory in bytes.
        device_alloc (int): Device-allocated memory in bytes.
        device_free (int): Device-free memory in bytes.

    Returns:
        None

    Notes:
        - It logs a memory summary, saves an allocated state snapshot, and visualizes memory usage.
    """
    if device is not None and logger is not None:
        logger.critical(torch.cuda.memory_summary(device))
    # snapshot right after an OOM happened
    print("saving allocated state during OOM")
    print("Tips: \nReduce sw_batch_size if there is an OOM (maybe even roi_size)")
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open(f"{output_dir}/oom_snapshot.pickle", "wb"))
    torch.cuda.memory._save_memory_usage(filename=f"{output_dir}/memory.svg", snapshot=snapshot)
    torch.cuda.memory._save_segment_usage(filename=f"{output_dir}/segments.svg", snapshot=snapshot)
