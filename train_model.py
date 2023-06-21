import torch
import numpy as np

from models.RFDTAN import RFDTAN
from models.trainer import Trainer

from helper.UCR_loader import get_UCR_data
from tsai.data.external import get_UCR_univariate_list


from config import get_argparser
from helper.NCC import NCC_pipeline
from helper.plotting_torch import plot_signals


def run_UCR_alignment(args):
    """
    Run an example of the full training pipline for DTAN on a UCR dataset.

    - Plots alignment, within class, for train and test set.

    Args:
        args: described at argparser.

    """
    # Data
    val_split = 0 if args.no_validation else 0.2
    train_loader, val_loader, test_loader = get_UCR_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_split=val_split,
        data_dir=args.data_dir,
        swap_channel_dim=True,
    )


    classes = torch.unique(train_loader.dataset[:][1])
    n_classes = len(classes)
    channels, input_shape = train_loader.dataset[0][0].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = RFDTAN(
        input_shape,
        channels,
        tess=args.tess_size,
        n_recurrence=args.n_recurrences,
        zero_boundary=args.zero_boundary,
        device=device,
        backbone=args.backbone,
    )


    # Train
    trainer_ucr = Trainer(
        exp_name = "my_exp_name",
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        device=device,
        args = args)

    trainer_ucr.train(print_model=True)
    
    # Evaluate
    res_dict = NCC_pipeline(model, train_loader, test_loader, args)

    # Plot alignment
    plot_signals(
        model, device=device, dataset_name=args.dataset, data_dir=args.data_dir
        )


    torch.cuda.empty_cache()

if __name__ == "__main__":
    BACKBONES = ['InceptionTime']
    DATASETS = ['ECGFiveDays']
    override_config = False
    # ALL datasets: 
    # DATASETS = get_UCR_univariate_list()
    for backbone in BACKBONES:
        for dataset in DATASETS:
            parser = get_argparser()
            args = parser.parse_args()
            # Overwrite args from the config.py file

            if override_config:
                args.backbone = backbone
                args.dataset = dataset
                args.ICAE_loss = True
                args.ICAE_triplets_loss = False
                args.WCSS_loss = False
                args.WCSS_triplets_loss = False
                args.smoothness_prior = False

                args.n_recurrences = 4
                args.batch_size = 128
                args.n_epochs = 500
                args.tess_size = 16
                args.lr = 0.0005
                args.no_validation = True



            run_UCR_alignment(args)


