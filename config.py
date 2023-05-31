import argparse


def get_argparser():
    parser = argparse.ArgumentParser()
    
    #############
    ### UTILS ###
    #############
    parser.add_argument(
        "--load_best",
        default=False,
        help="Load best model based on validation loss",
        action="store_true",
    )
    parser.add_argument("--no_validation", default=False, action="store_true")

    # The dataset
    parser.add_argument("--dataset", type=str, default="ECGFiveDays")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./data/checkpoints/",
    )
    ###############
    ### LOSSES ####
    ###############
    parser.add_argument(
        "--loss_weights",
        default={
            "WCSS_loss": 1.0,
            "WCSS_triplets_loss": 1.0,
            "prior_loss": 1.0,
            "ICAE_loss": 1,
            "ICAE_triplets_loss": 1,
        },
        help="A dictonary containing the weighted sum of all losses",
    )

    ### WCSS ###
    parser.add_argument(
        "--WCSS_loss",
        default=False,
        help="within-class variance flag",
        action="store_true",
    )
    parser.add_argument(
        "--WCSS_triplets_loss",
        default=False,
        help="within-class variance flag",
        action="store_true",
    )
    ### PRIORS ####
    parser.add_argument(
        "--smoothness_prior",
        default=False,
        help="smoothness prior flag",
        action="store_true",
    )

    parser.add_argument(
        "--lambda_smooth",
        type=float,
        default=0.5,
        help="lambda_smooth, larger values -> smoother warps",
    )
    parser.add_argument(
        "--lambda_var",
        type=float,
        default=0.01,
        help="lambda_var, larger values -> larger warps",
    )
    
    
    ### ICAE and triplet ###
    
    parser.add_argument(
        "--ICAE_loss",
        default=False,
        help="Inverse Consistency Averaging Error",
        action="store_true",
    )
    parser.add_argument(
        "--ICAE_triplets_loss",
        default=False,
        help="Inverse Consistency Averaging Error + triplets loss",
        action="store_true",
    )

    #####################
    ### DTAN and CPAB ###
    #####################

    parser.add_argument(
        "--tess_size", type=int, default=16, help="CPA velocity field partition"
    )
    parser.add_argument(
        "--zero_boundary", type=bool, default=True, help="zero boundary constrain"
    )

    parser.add_argument(
        "--n_recurrences", type=int, default=4, help="number of recurrences of R-DTAN"
    )


    parser.add_argument("--backbone", type=str, default="InceptionTime")

    ################
    ### Training ###
    ################
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")

    parser.add_argument("--lr_step", type=float, default=0.5, help="drop lr by K")

    parser.add_argument("--exp_name", type=str, default="default_name")

    return parser
