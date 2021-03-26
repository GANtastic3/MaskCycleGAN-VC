"""
Arguments for training ASR model.
Inherits BaseArgParser.
"""

from args.train_arg_parser import TrainArgParser


class ASRTrainArgParser(TrainArgParser):
    """
    Class which implements an argument parser for args used only in train mode.
    It inherits BaseArgParser.
    """

    def __init__(self):
        super(ASRTrainArgParser, self).__init__()
        self.isTrain = True

        self.parser.add_argument(
            '--dropout', type=float, default=0.1, help='Dropout rate.')
        self.parser.add_argument(
            '--gamma', type=float, default=0.99, help='Annealing rate for LR scheduler.')

        # Model args
        self.parser.add_argument(
            '--n_cnn_layers', type=int, default=3, help='Numer of CNN layers.')
        self.parser.add_argument(
            '--n_rnn_layers', type=int, default=5, help='Number of RNN layers')
        self.parser.add_argument(
            '--rnn_dim', type=int, default=512, help='Dimensionality of RNN')
        self.parser.add_argument(
            '--n_class', type=int, default=29, help='Number of output classes.')
        self.parser.add_argument(
            '--n_feats', type=int, default=128, help='Number of features.')
        self.parser.add_argument(
            '--stride', type=int, default=2, help='Conv2D kernel stride.')

        self.parser.add_argument(
            '--pretrained_ckpt_path', type=str,
            default=None,
            help='Model pretrained on Librispeech.')
        self.parser.add_argument('--librispeech', default=False, action='store_true',
                                 help=('Train with libirspeech dataset.'))
