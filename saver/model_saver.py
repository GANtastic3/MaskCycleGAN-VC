""" Saves and loads models from checkpoints. """

import os
import shutil
import torch


class ModelSaver(object):
    """
    Class to save and load model ckpts.
    Attributes:
        ckpt_dir (str): Directory to save checkpoints.
        max_ckpts (int): Maximum number of checkpoints to keep before overwriting old ones
        ckpt_names(list(str)): Names of ckpts stored in ckpt_dir
        metric_name (str): Name of metric used to determine best model
        maximize_metric (bool): If true, best checkpoint is that which maximizes the metric value passed in via save
            If false, best checkpoint minimizes the metric
        best_metric_val (float): Value of the best metric
        load_epoch (int): Epoch from which model was loaded from
    Methods:
        get_last_saved_epoch(): Returns the last epoch at which a checkpoint was saved
        _is_best(self, metric_val): Check whether metric_val is the best one we've seen so far
        save(epoch, model, lr_scheduler, optimizer, device, model_name, metric_val):
            If step corresponds to a save step, save model parameters to disk.
        load_model(self, model, model_name=None, ckpt_path=None, optimizer=None, scheduler=None):
            Function that loads model, optimizer, and scheduler statedicts froma ckpt.
    """

    def __init__(self, args, max_ckpts=None, metric_name=None, maximize_metric=False):
        """
        Args:
            ckpt_dir (str): Directory to save checkpoints.
            max_ckpts (:obj:`int`, optional): Maximum number of checkpoints to keep before overwriting old ones.
            metric_name (:obj:`str`, optional): Name of metric used to determine best model.
            maximize_metric: If true, best checkpoint is that which maximizes the metric value passed in via save.
                If false, best checkpoint minimizes the metric.
        """
        super(ModelSaver, self).__init__()

        self.args = args
        self.ckpt_dir = args.ckpt_dir
        self.max_ckpts = max_ckpts
        self.ckpt_names = sorted([name for name in os.listdir(
            self.ckpt_dir) if name.split(".", 1)[1] == "pth.tar"])

    def save(self, epoch, model, optimizer, lr_scheduler, device, model_name):
        """If this step corresponds to a save step, save model parameters to disk.

        Args:
            epoch (int): Current epoch
            model: Model to save
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler for optimizer
            optimizer (torch.optim.Optimizer): Optimizer for model parameters
            device (str): Device where the model/optimizer parameters belong
            model_name (str): Name of model to save
        """
        # Unwrap data parallel module if needed
        try:
            model_class = model.module.__class__.__name__
            model_state = model.to('cpu').module.state_dict()
            print("Saving unwrapped DataParallel module.")
        except AttributeError:
            model_class = model.__class__.__name__
            model_state = model.to('cpu').state_dict()

        ckpt_dict = {
            'ckpt_info': {'epoch': epoch},
            'model_class': model_class,
            'model_state': model_state,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        }

        model.to(device)

        file_name = f'{str(epoch).zfill(5)}_{model_name}.pth.tar'

        ckpt_path = os.path.join(self.ckpt_dir, file_name)
        torch.save(ckpt_dict, ckpt_path)
        print(f"Saved model to {ckpt_path}")

        # Remove a checkpoint if more than max_ckpts ckpts saved
        if self.max_ckpts:
            self.ckpt_names.append(ckpt_path)
            if len(self.ckpt_names) > self.max_ckpts:
                oldest_ckpt = os.path.join(
                    self.ckpt_dir, self.ckpt_names.pop(0))
                os.remove(oldest_ckpt)
                print(
                    f"Exceeded max number of checkpoints so deleting {oldest_ckpt}")

    def load_model(self, model, model_name=None, ckpt_path=None, optimizer=None, scheduler=None):
        """
        Function that loads model, optimizer, and scheduler statedicts from a ckpt.
        If ckpt_path is specified, loads model from the path.
        If there is a best ckpt in the ckpt directory, it loads the best model.
        Else it loads the most recent model.
        Args:
            model (nn.Module): Initialized model objects
            model_name (:obj:`str`, optional): Name of model to load
            ckpt_path (:obj:`str`, optional): Path to saved model ckpt
            optimizer (:obj:`torch.optim.Optimizer`, optional): Initialized optimizer object
            scheduler (:obj:`torch.optim.lr_scheduler`, optional): Initilazied scheduler object
        """
        ckpt_paths = sorted([name for name in os.listdir(
            self.ckpt_dir) if name.split(".", 1)[1] == "pth.tar"])

        if ckpt_path is None:
            if model_name and hasattr(self.args, 'load_epoch'):
                file_name = f'{str(self.args.load_epoch).zfill(5)}_{model_name}.pth.tar'
                ckpt_path = os.path.join(self.ckpt_dir, file_name)
            else:
                print("No checkpoint found. Failed to load load model checkpoint.")
                return

        checkpoint = torch.load(ckpt_path, map_location=self.args.gpu_ids[0])
        model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print(f"Loaded {checkpoint['model_class']} from {ckpt_path}")
