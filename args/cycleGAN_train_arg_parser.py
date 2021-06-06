"""
Arguments for MaskCycleGAN-VC training.
Inherits TrainArgParser.
"""

from args.train_arg_parser import TrainArgParser


class CycleGANTrainArgParser(TrainArgParser):
    """
    Class which implements an argument parser for args used only in training MaskCycleGAN-VC.
    It inherits TrainArgParser.
    """

    def __init__(self):
        super(CycleGANTrainArgParser, self).__init__()

        self.parser.add_argument('--sample_rate', type=int, default=22050, help='Sampling rate of mel-spectrograms.')
        self.parser.add_argument(
            '--speaker_A_id', type=str, default="28", help='Source speaker id (From VOC dataset).')
        self.parser.add_argument(
            '--speaker_B_id', type=str, default="DCB_se2_ag3_m_02_1", help='Target speaker id (From CORAAL dataset).')
        self.parser.add_argument(
            '--preprocessed_data_dir', type=str, default="vcc2018_training_preprocessed/", help='Directory containing preprocessed dataset files.')

        # Model args
        self.parser.add_argument(
            '--generator_lr', type=float, default=2e-4, help='Initial generator learning rate.')
        self.parser.add_argument(
            '--discriminator_lr', type=float, default=1e-4, help='Initial discrminator learning rate.')
        
        # Loss lambdas
        self.parser.add_argument(
            '--cycle_loss_lambda', type=float, default=10, help='Lambda value for cycle consistency loss.')
        self.parser.add_argument(
            '--identity_loss_lambda', type=float, default=5, help='Lambda value for identity loss.')
        
        self.parser.add_argument(
            '--epochs_per_plot', type=int, default=2, help='Epochs per save plot.')
        
        self.parser.add_argument(
            '--num_frames', type=int, default=64, help='Num frames per training sample.'
        )
        self.parser.add_argument(
            '--num_frames_validation', type=int, default=320, help='Num frames per validation sample.'
        )
        self.parser.add_argument(
            '--max_mask_len', type=int, default=32, help='Maximum length of mask for Mask-CycleGAN-VC.'
        )

        self.parser.set_defaults(batch_size=1, num_epochs=50, decay_after=1e4, start_epoch=1, steps_per_print=100, num_frames=64)