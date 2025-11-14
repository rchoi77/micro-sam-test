import os
import wandb
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch_em.trainer.logger_base import TorchEmLogger
from torchvision.utils import make_grid
from torch_em.trainer.tensorboard_logger import normalize_im

class WandBJointSamLogger(TorchEmLogger):
    """
    Custom logger that replicates the class JointSamLogger but adds WandB logging as well.
    
    Initialize WandB outside.
    """

    def __init__(self, trainer, save_root, wandb_train_log_interval=10, wandb_run=None, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.log_dir = f"./logs/{trainer.name}" if save_root is None else\
            os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval
        self.wandb_train_log_interval = wandb_train_log_interval
        self.wandb_run = wandb_run
        """
        The logger class is instantiated in a method from an ancestor class (likely torch_em.DefaultTrainer).

        Set wand_image_log_interval and wandb_train_log_interval to change the frequency of wandb logs.

        Not a hard science, but past training runs have been ~2000 iterations (steps), and
        JointSamTrainer does a training log per iteration. So, setting wandb_train_log_interval=10
        gets around 200 points per training run. With a sweep, you run multiple training runs. So,
        in the interest of a non-laggy dashboard, try to limit total points to < 10,000 (?)
        """

    def add_image(self, x, y, samples, name, step):
        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]

        image = normalize_im(x[selection].cpu())

        self.tb.add_image(tag=f"{name}/input", img_tensor=image, global_step=step)
        self.tb.add_image(tag=f"{name}/target", img_tensor=y[selection], global_step=step)
        sample_grid = make_grid([sample[0] for sample in samples], nrow=4, padding=4)
        self.tb.add_image(tag=f"{name}/samples", img_tensor=sample_grid, global_step=step)

        # Custom WandB image log
        if self.wandb_run:
            self.wandb_run.log({
                f"{name}/input": wandb.Image(image),
                f"{name}/target": wandb.Image(y[selection]),
                f"{name}/samples": wandb.Image(sample_grid),
            }, step=step)

    def log_train(
        self, step, loss, lr, x, y, samples, mask_loss, iou_regression_loss, model_iou, instance_loss
    ):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/mask_loss", scalar_value=mask_loss, global_step=step)
        self.tb.add_scalar(tag="train/iou_loss", scalar_value=iou_regression_loss, global_step=step)
        self.tb.add_scalar(tag="train/model_iou", scalar_value=model_iou, global_step=step)
        self.tb.add_scalar(tag="train/instance_loss", scalar_value=instance_loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)

        # Custom WandB training log
        if self.wandb_run:
            if step % self.wandb_train_log_interval == 0:
                self.wandb_run.log({
                    "train/loss": loss,
                    "train/mask_loss": mask_loss,
                    "train/iou_loss": iou_regression_loss,
                    "train/model_iou": model_iou,
                    "train/instance_loss": instance_loss,
                    "train/learning_rate": lr,
                }, step=step)

        if step % self.log_image_interval == 0:
            self.add_image(x, y, samples, "train", step)

    def log_validation(
        self, step, metric, loss, x, y, samples, mask_loss, iou_regression_loss, model_iou, instance_loss
    ):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/mask_loss", scalar_value=mask_loss, global_step=step)
        self.tb.add_scalar(tag="validation/iou_loss", scalar_value=iou_regression_loss, global_step=step)
        self.tb.add_scalar(tag="validation/model_iou", scalar_value=model_iou, global_step=step)
        self.tb.add_scalar(tag="train/instance_loss", scalar_value=instance_loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)

        # Custom WandB validation log
        if self.wandb_run:
            self.wandb_run.log({
                "validation/loss": loss,
                "validation/mask_loss": mask_loss,
                "validation/iou_loss": iou_regression_loss,
                "validation/model_iou": model_iou,
                "validation/metric": metric,
            }, step=step)

        self.add_image(x, y, samples, "validation", step)