""" Logs metrics and metadata to visualize on tensorboard """

import os
from datetime import datetime
from tensorboardX import SummaryWriter


class BaseLogger(object):
    """
    Class which implements a Logger.
    Attributes:
        args : Namespace Program arguments
        batch_size (int): Batch size
        dataset_len (int): Number of samples in the dataset
        save_dir (str): Save directory path
        summary_writer (tensorboardX.SummaryWriter): Writes entires to event files in args.log_dir
        log_path (str): Path to .log file <- not the same as the event file
        epoch (int): Current epoch
        iter (int): Current iteration within epoch
        global_step (int): Current iteration overall
    Methods:
        _log_text(text_dict): Log all strings in a dict as scalars to TensorBoard
        _log_scalars(scalar_dict, print_to_stdout): Log all values in a dict as scalars to TensorBoard
        write(message, print_to_stdout): Write a message to the .log file. If print_to_stdout is True, also print to stdout
        start_iter(): Log info for start of an iteration
        end_iter(): Log info for end of an iteration
        start_epoch(): Log info for start of an epoch
        end_epoch(metrics): Log info for end of an epoch. Save model parameters and update learning rate
    """

    def __init__(self, args, dataset_len):
        """
        Args:
            args (Namespace): Program arguments
            dataset_len (int): Number of samples in dataset
        """

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.args = args
        self.batch_size = args.batch_size
        self.dataset_len = dataset_len
        self.save_dir = args.save_dir

        # log_dir, is the directory for tensorboard logs: hpylori/logs/
        log_dir = os.path.join(
            args.save_dir, 'logs', args.name + '_' + datetime.now().strftime('%y%m%d_%H%M%S'))
        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.log_path = os.path.join(
            self.save_dir, args.name, '{}.log'.format(args.name))
        self.epoch = args.start_epoch
        self.iter = 0
        self.global_step = round_down(
            (self.epoch - 1) * dataset_len, args.batch_size)

    def log_text(self, text_dict):
        """
        Log all strings in a dict as scalars to TensorBoard.
        Args:
            text_dict (dict): str to str dictionary
        """
        for k, v in text_dict.items():
            self.summary_writer.add_text(k, str(v), self.global_step)

    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """
        Log all values in a dict as scalars to TensorBoard.
        Args:
            scalar_dict (dict): str to scalar dictionary
            print_to_stdout (bool): If True, print scalars to stdout
        """
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {}]'.format(k, v))
            # Group in TensorBoard by split. eg. (D_A, D_B) are grouped, (G_A, G_B) are grouped
            k = k.replace('_', '/')
            self.summary_writer.add_scalar(k, v, self.global_step)

    def write(self, message, print_to_stdout=True):
        """
        Write a message to the .log file. If print_to_stdout is True, also print to stdout.
        Args:
            message (str): Message to write to .log (and stdout if applicable)
            print_to_stdout (bool): If True, print message to stdout
        """
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics):
        """
        Log info for end of an epoch. Save model parameters and update learning rate.
        Args:
        metrics (dict): str to scalar dictionary containing metrics such as losses to log
        """
        raise NotImplementedError
