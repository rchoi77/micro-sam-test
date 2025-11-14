import os
import numpy as np

# TODO: if we remove raw_key, label_key, and other key parameters, we can remove this dep 
from glob import glob

from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, random_split

import torch_em
from torch_em.util import load_data
from wandb.sdk import wandb_run

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from ..util import get_model_names

import wandb
from wandb_logger import WandBJointSamLogger

FilePath = Union[str, os.PathLike]

# IMPORTANT: run this script directly, unexpected behavior on import

# Config for 
sweep_config = {
    "method": "random",
    "metric": {
        "goal": "minimize",
        "name": "validation/metric"
    },
    "parameters": {
        "n_objects_per_batch": {
            "values": [10, 25, 50]
        },
        "patch_shape": {
            "values": [(512, 512), (1024, 1024)]
        },
        "batch_size": {
            "values": [1, 2]
        },
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-4
        },
    },
}

def require_8bit(x):
    """Transformation to require 8bit input data range (0-255).
    """
    if x.max() < 1:
        x = x * 255
    return x

def default_sam_dataset(
    raw_paths: Union[List[FilePath], FilePath],
    raw_key: Optional[str],
    label_paths: Union[List[FilePath], FilePath],
    label_key: Optional[str],
    patch_shape: Tuple[int],
    with_segmentation_decoder: bool,
    with_channels: Optional[bool] = None,
    train_instance_segmentation_only: bool = False,
    sampler: Optional[Callable] = None,
    raw_transform: Optional[Callable] = None,
    n_samples: Optional[int] = None,
    is_train: bool = True,
    min_size: int = 25,
    max_sampling_attempts: Optional[int] = None,
    rois: Optional[Union[slice, Tuple[slice, ...]]] = None,
    is_multi_tensor: bool = True,
    **kwargs,
) -> Dataset:
    """Create a PyTorch Dataset for training a SAM model.

    Args:
        raw_paths: The path(s) to the image data used for training.
            Can either be multiple 2D images or volumetric data.
        raw_key: The key for accessing the image data. Internal filepath for hdf5-like input
            or a glob pattern for selecting multiple files.
        label_paths: The path(s) to the label data used for training.
            Can either be multiple 2D images or volumetric data.
        label_key: The key for accessing the label data. Internal filepath for hdf5-like input
            or a glob pattern for selecting multiple files.
        patch_shape: The shape for training patches.
        with_segmentation_decoder: Whether to train with additional segmentation decoder.
        with_channels: Whether the image data has channels. By default, it makes the decision based on inputs.
        train_instance_segmentation_only: Set this argument to True in order to
            pass the dataset to `train_instance_segmentation`. By default, set to 'False'.
        sampler: A sampler to reject batches according to a given criterion.
        raw_transform: Transformation applied to the image data.
            If not given the data will be cast to 8bit.
        n_samples: The number of samples for this dataset.
        is_train: Whether this dataset is used for training or validation. By default, set to 'True'.
        min_size: Minimal object size. Smaller objects will be filtered. By default, set to '25'.
        max_sampling_attempts: Number of sampling attempts to make from a dataset.
        rois: The region of interest(s) for the data.
        is_multi_tensor: Whether the input data to data transforms is multiple tensors or not.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """

    # Check if this dataset should be used for instance segmentation only training.
    # If yes, we set return_instances to False, since the instance channel must not
    # be passed for this training mode.
    return_instances = True
    if train_instance_segmentation_only:
        if not with_segmentation_decoder:
            raise ValueError(
                "If 'train_instance_segmentation_only' is True, then 'with_segmentation_decoder' must also be True."
            )
        return_instances = False

    # If a sampler is not passed, then we set a MinInstanceSampler, which requires 3 distinct instances per sample.
    # This is necessary, because training for interactive segmentation does not work on 'empty' images.
    # However, if we train only the automatic instance segmentation decoder, then this sampler is not required
    # and we do not set a default sampler.
    if sampler is None and not train_instance_segmentation_only:
        sampler = torch_em.data.sampler.MinInstanceSampler(2, min_size=min_size)

    # By default, let the 'default_segmentation_dataset' heuristic decide for itself.
    is_seg_dataset = kwargs.pop("is_seg_dataset", None)

    # Check if the raw inputs are RGB or not. If yes, use 'ImageCollectionDataset'.
    # Get valid raw paths to make checks possible.
    if raw_key and "*" in raw_key:  # Use the wildcard pattern to find the filepath to only one image.
        rpath = glob(os.path.join(raw_paths if isinstance(raw_paths, str) else raw_paths[0], raw_key))[0]
    else:  # Otherwise, either 'raw_key' is None or container format, supported by 'elf', then we load 1 filepath.
        rpath = raw_paths if isinstance(raw_paths, str) else raw_paths[0]

    # Load one of the raw inputs to validate whether it is RGB or not.
    test_raw_inputs = load_data(path=rpath, key=raw_key if raw_key and "*" not in raw_key else None)
    if test_raw_inputs.ndim == 3:
        if test_raw_inputs.shape[-1] == 3:  # i.e. if it is an RGB image and has channels last.
            is_seg_dataset = False  # we use 'ImageCollectionDataset' in this case.
            # We need to provide a list of inputs to 'ImageCollectionDataset'.
            raw_paths = [raw_paths] if isinstance(raw_paths, str) else raw_paths
            label_paths = [label_paths] if isinstance(label_paths, str) else label_paths

            # This is not relevant for 'ImageCollectionDataset'. Hence, we set 'with_channels' to 'False'.
            with_channels = False if with_channels is None else with_channels

        elif test_raw_inputs.shape[0] == 3:  # i.e. if it is a RGB image and has 3 channels first.
            # This is relevant for 'SegmentationDataset'. If not provided by the user, we set this to 'True'.
            with_channels = True if with_channels is None else with_channels

    # Set 'with_channels' to 'False', i.e. the default behavior of 'default_segmentation_dataset'
    # Otherwise, let the user make the choice as priority, else set this to our suggested default.
    with_channels = False if with_channels is None else with_channels

    # Set the data transformations.
    if raw_transform is None:
        raw_transform = require_8bit

    # Prepare the label transform.
    if with_segmentation_decoder:
        default_label_transform = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=return_instances,
            min_size=min_size,
        )
    else:
        default_label_transform = torch_em.transform.label.MinSizeLabelTransform(min_size=min_size)

    # Allow combining label transforms.
    custom_label_transform = kwargs.pop("label_transform", None)
    if custom_label_transform is None:
        label_transform = default_label_transform
    else:
        label_transform = torch_em.transform.generic.Compose(
            custom_label_transform, default_label_transform, is_multi_tensor=is_multi_tensor
        )

    # Set a minimum number of samples per epoch.
    if n_samples is None:
        loader = torch_em.default_segmentation_loader(
            raw_paths=raw_paths,
            raw_key=raw_key,
            label_paths=label_paths,
            label_key=label_key,
            batch_size=1,
            patch_shape=patch_shape,
            with_channels=with_channels,
            ndim=2,
            is_seg_dataset=is_seg_dataset,
            raw_transform=raw_transform,
            rois=rois,
            **kwargs
        )
        n_samples = max(len(loader), 100 if is_train else 5)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=raw_key,
        label_paths=label_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        label_transform=label_transform,
        with_channels=with_channels,
        ndim=2,
        sampler=sampler,
        n_samples=n_samples,
        is_seg_dataset=is_seg_dataset,
        rois=rois,
        **kwargs,
    )

    if max_sampling_attempts is not None:
        if isinstance(dataset, torch_em.data.concat_dataset.ConcatDataset):
            for ds in dataset.datasets:
                ds.max_sampling_attempts = max_sampling_attempts
        else:
            dataset.max_sampling_attempts = max_sampling_attempts

    return dataset

def get_dataloaders(
    image_paths: Union[List[FilePath], FilePath],
    label_paths: Union[List[FilePath], FilePath],
    patch_shape: Tuple[int],
    batch_size: int,
    train_instance_segmentation: bool,
    val_paths: Optional[Union[List[FilePath], FilePath]] = None,
    val_label_paths: Optional[Union[List[FilePath], FilePath]] = None,
    preprocessor: Optional[Callable] = None,
    n_samples: Optional[int] = None,
) -> DataLoader:
    """Return train and val dataloader for finetuning SAM.

    The data loader must be a torch data loader that returns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive

    Args:
        image_paths: Training image paths.
        label_paths: Label image paths.
        patch_shape: The shape of patches to be used in training.
        batch_size: 
        train_instance_segmentation: Boolean for whether to train the additional segmentation decoder.
        val_paths: Validation set image paths.
        val_label_paths: Validation set label paths.
        preprocessor: Raw transform function to apply to images as preprocessing.
        n_samples: Number of samples in the dataset. Leave as None for now.

    Returns:
        Two dataloaders, one for the training set, one for the validation set.
    """
    # Get datasets
    dataset = default_sam_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        with_segmentation_decoder=train_instance_segmentation,
        raw_transform=preprocessor,
        n_samples=n_samples,
    )

    # If validation images are not provided, create a val split from the training data.
    if val_paths is None:
        assert val_label_paths is None
        # Use 10% of the dataset for validation - at least one image - for validation.
        n_val = max(1, int(0.1 * len(dataset)))
        train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - n_val, n_val])
    else:  # If val images provided, we create a new dataset for it.
        train_dataset = dataset
        val_dataset = default_sam_dataset(
            raw_paths=val_paths,
            raw_key=None,
            label_paths=val_label_paths,
            label_key=None,
            patch_shape=patch_shape,
            with_segmentation_decoder=train_instance_segmentation,
            raw_transform=preprocessor,
        )
    
    # Get dataloaders
    train_loader = torch_em.get_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch_em.get_data_loader(val_dataset, batch_size=1, shuffle=True)

    return train_loader, val_loader

    # Leaving this here as a note. Mainly, why torch_em.transform.label.connected_components?
    # And, what is the difference between torch_em.get_data_loader and torch_em.default_segmentation_loader?
    """
    if train_instance_segmentation:
        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components

    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir,
        raw_key=raw_key,
        label_paths=segmentation_dir,
        label_key=label_key,
        patch_shape=patch_shape, #hparam
        batch_size=batch_size, #hparam
        ndim=2,
        is_seg_dataset=True,
        label_transform=label_transform,
        raw_transform=sam_training.identity,
    )
    """


def run_training(
    image_paths: Union[List[FilePath], FilePath],
    label_paths: Union[List[FilePath], FilePath],
    checkpoint_name: str,
    model_type: str,
    train_instance_segmentation: bool,
    output_path: Union[os.PathLike, str],
    wandb_project_name: str,
    wandb_run_notes: str,
    val_paths: Optional[Union[List[FilePath], FilePath]] = None,
    val_label_paths: Optional[Union[List[FilePath], FilePath]] = None,
    batch_size: int = 1,
    patch_shape: Tuple[int, int] = (512, 512),
    n_objects_per_batch: int = 25,
    lr: float = 1e-5,
    n_samples: Optional[int] = None,
    wandb_config: Optional[dict] = None,
) -> None:
    """Run the actual model training.

    Args:
        image_paths: Training image paths.
        label_paths: Label image paths.
        checkpoint_name: Name for the checkpoint.
        model_type: SAM model type (vit_b, vit_l, etc.).
        train_instance_segmentation: Whether to train with segmentation decoder.
        output_path: Directory to save outputs.
        val_paths: Validation image paths.
        val_label_paths: Validation label paths.
        batch_size: Number of images per training batch.
        patch_shape: Size of training patches.
        n_objects_per_batch: Number of objects sampled per batch.
        lr: Learning rate.
        n_samples: Number of samples per dataset (None = auto).
        wandb_config: Optional config for WandB used when not using a sweep.
    """
    if wandb_config:
        run = wandb.init(
            project=wandb_project_name,
            notes=wandb_run_notes,
            config=wandb_config,
        )
    else:
        # sweep, config auto-injected by agent
        run = wandb.init(
            project=wandb_project_name,
            notes=wandb_run_notes,
        )

    # TODO: Preprocessing
    if True:
        _raw_transform = sam_training.identity
        #_raw_transform = get_raw_transform(args.preprocess)

    # Get the dataloaders.
    train_loader, val_loader = get_dataloaders(
        image_paths=image_paths,
        label_paths=label_paths,
        train_instance_segmentation=train_instance_segmentation,
        val_paths=val_paths,
        val_label_paths=val_label_paths,
        preprocessor=_raw_transform,
        patch_shape=patch_shape,
        batch_size=batch_size,
        n_samples=n_samples,
    )

    # Run training.
    device = torch.device("cuda")  # The device used for training. Notably, could be 'mps' for mac. TODO: automatic device detection
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        with_segmentation_decoder=train_instance_segmentation,
        save_root=output_path,
        logger=WandBJointSamLogger,
        logger_kwargs={'wandb_run': run},
        device=device,
        n_objects_per_batch=n_objects_per_batch,
        lr=lr,
    )

    run.finish()

    # All hyperparameters for training. See MicroSAM documentation for hardware limitations.
    # dataloader params
    #batch_size = 1  # The number of images in the training batch. Total samples is batch_size * n_objects_per_batch
    #patch_shape = (512, 512)  # The size of patches for training
    #n_samples = None # The number of samples per dataset (leave as None unless you have a reason to do otherwise, resolves by - default)
    # training params
    #n_objects_per_batch = 25  # The number of objects per batch that will be sampled
    #lr


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    # export the model after training so that it can be used by the rest of the 'micro_sam' library
    export_path = "./hello.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def main():
    """Finetune a Segment Anything model.

    This example uses image data and segmentations from the cell tracking challenge,
    but can easily be adapted for other data (including data you have annotated with micro_sam beforehand).
    """
    import argparse
    parser = argparse.ArgumentParser(description="Custom training script with WandB logging. Provides a sweep option.")
    #####################################
    ########## TRAINING INPUTS ##########
    #####################################
    parser.add_argument(
        "--images", required=True, type=str, nargs="*",
        help="Filepath(s) to input images. E.g. 'trainingdata/images/*.tiff'"
    )
    parser.add_argument(
        "--labels", required=True, type=str, nargs="*",
        help="Filepath(s) to ground-truth labels. Please match the --images parameter."
    )
    parser.add_argument(
        "--val-images", type=str, nargs="*",
        help="Filepath to images for validation or the directory where the image data is stored."
    )
    parser.add_argument(
        "--val-labels", type=str, nargs="*",
        help="Filepath to ground-truth labels for validation or the directory where the label data is stored."
    )
    ############################################
    ########## GENERAL MODEL SETTINGS ##########
    ############################################
    # TODO: get model list, resolve dependencies first
    parser.add_argument(
        "-m", "--model-type", type=str, default=None,
        help=f"The Segment Anything Model whose weights will be initialized for fine-tuning. For this script, one of vit_b, vit_l. Other model support can be easily added."
    )
    # TODO: make these two non-required arguments with a defaults
    parser.add_argument(
        "-c", "--checkpoint-name", required=True, type=str,
        help="The name of the resulting fine-tuned model for your identification purposes. See --output-path."
    )
    parser.add_argument(
        "-o", "--output-path", required=True, type=str,
        help="The directory to store the fine-tuned model. E.g. stores to ./<OUTPUT_PATH>/checkpoints/<checkpoint_name>/best.pt"
    )
    parser.add_argument(
        "--sweep", action="store_true", default=False, help="Whether to do a WandB sweep over hyperparameter space."
    )
    parser.add_argument(
        "-n", "--note", type=str, default="No notes provided.",
        help=f"Note for the WandB run."
    )

    args = parser.parse_args()

    ############################################
    ############################################
    ############################################

    model_type = args.model_type
    checkpoint_name = args.checkpoint_name
    output_path = args.output_path

    train_images, train_labels, val_images, val_labels = args.images, args.labels, args.val_images, args.val_labels

    # Handle base model choice
    available_models = list(get_model_names())
    available_models = ", ".join(available_models)
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"'{model_type}' is not a valid choice of model.")
    elif model_type is None:
        model_type = "vit_b"
    assert(model_type in ["vit_b", "vit_l"]) # TODO: allow other options; but, these are probably all we need for our use case, maybe vit_b_lm or a pathosam model

    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = True

    # WandB project settings
    wandb_project_name = "microsam-sweep"
    wandb_run_notes = args.note
    if not args.sweep:
        # Run one training run with WandB logging
        config = {
            "lr": 0.01,
        }
        
        run_training(
            image_paths=train_images,
            label_paths=train_labels,
            checkpoint_name=checkpoint_name,
            model_type=model_type,
            train_instance_segmentation=train_instance_segmentation,
            output_path=output_path,
            val_paths=val_images,
            val_label_paths=val_labels,
            wandb_project_name=wandb_project_name,
            wandb_run_notes=wandb_run_notes,
            wandb_config=config,
        )
    else:
        # Run a WandB sweep over hyperparameter space
        # Uses config defined at top of file
        def sweep_train():
            """Training function for WandB sweep - gets hyperparameters from sweep agent."""
            run_training(
                image_paths=train_images,
                label_paths=train_labels,
                checkpoint_name=checkpoint_name,
                model_type=model_type,
                train_instance_segmentation=train_instance_segmentation,
                output_path=output_path,
                val_paths=val_images,
                val_label_paths=val_labels,
                wandb_project_name=wandb_project_name,
                wandb_run_notes=wandb_run_notes,
            )

        # Initialize sweep and run agent
        sweep_id = wandb.sweep(sweep=sweep_config, project="microsam-sweep")
        wandb.agent(sweep_id, function=sweep_train, count=10) # TODO: possibly change count to a user input

        #print(f"{'='*70}")
        #print(f"SWEEP COMPLETE!")
        # TODO: get best results from wandb api
        #api.Sweep('...').best_run.config
        #print(f"Best set of hyperparams: {}")
        #print(f"{'='*70}\n")

    # May want to export model for ???
    #export_model(checkpoint_name, model_type)

if __name__ == "__main__":
    main()
