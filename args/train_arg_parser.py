"""
Arguments for training.
Inherits BaseArgParser.
"""

from args.base_arg_parser import BaseArgParser


class TrainArgParser(BaseArgParser):
    """
    Class which implements an argument parser for args used only in train mode.
    It inherits BaseArgParser.
    """

    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.isTrain = True

        self.parser.add_argument('--num_epochs', type=int, default=6500, help='Number of epochs to train.')
        self.parser.add_argument(
            '--decay_after', type=float, default=2e5, help='Decay learning rate after n iterations.')
        self.parser.add_argument(
            '--stop_identity_after', type=float, default=1e4, help='Stop using identity loss after n iterations.')

        self.parser.add_argument('--max_ckpts', type=int, default=3, help='Max ckpts to save.')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
