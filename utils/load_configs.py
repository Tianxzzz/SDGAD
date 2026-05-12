import argparse
import sys
import torch


def get_anomaly_detection_args(is_evaluation: bool = False):
    parser = argparse.ArgumentParser("Interface for the CTDGAD anomaly detection task")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikipedia",
        choices=[
            "wikipedia",
            "reddit",
            "mooc",
            "lastfm",
            "enron",
            "uci",
            "uci_mix_0.001",
            "lastfm_mix_0.001",
            "enron_mix_0.001",
        ],
        help="dataset to be used",
    )
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument(
        "--model_name",
        type=str,
        default="GeneralDyG",
        choices=[
            "JODIE",
            "DyRep",
            "TGAT",
            "TGN",
            "CAWN",
            "TCL",
            "SODA",
            "SODA_cat",
            "GraphMixer",
            "DyGFormer",
            "GeneralDyG",
        ],
        help="name of the model backbone",
    )
    parser.add_argument("--pos_beta", type=float, default=0.01, help="positives beta for logp boundary")
    parser.add_argument("--margin_tau", type=float, default=0.1, help="margin for logp boundary")
    parser.add_argument("--lamda1", type=float, default=1.0, help="weight for flow consistency loss")
    parser.add_argument("--lamda2", type=float, default=0.5, help="weight for bi-occurrence loss")
    parser.add_argument("--r_max", type=float, default=0.4, help="r_max for occ loss")
    parser.add_argument("--r_coff", type=float, default=0.99, help="r coefficient for occ loss")
    parser.add_argument("--few_shot_k", type=int, default=None, help="number of anomaly labels to keep during few-shot training")

    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--num_neighbors", type=int, default=20, help="number of neighbors for each node")
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="strategy for sampling historical neighbors",
    )
    parser.add_argument(
        "--time_scaling_factor",
        type=float,
        default=1e-6,
        help="scaling for time aware sampling (only works when time_interval_aware is used)",
    )
    parser.add_argument("--num_walk_heads", type=int, default=8, help="number of walk attention heads (CAWN)")
    parser.add_argument("--num_heads", type=int, default=2, help="number of attention heads in the model")
    parser.add_argument("--num_layers", type=int, default=2, help="number of model layers")
    parser.add_argument("--walk_length", type=int, default=1, help="length of each random walk (CAWN)")
    parser.add_argument("--time_gap", type=int, default=2000, help="time gap used for node history encoding")
    parser.add_argument("--time_feat_dim", type=int, default=100, help="time embedding dimension")
    parser.add_argument("--position_feat_dim", type=int, default=172, help="position embedding dimension (CAWN)")
    parser.add_argument("--patch_size", type=int, default=1, help="patch size (DyGFormer)")
    parser.add_argument("--channel_embedding_dim", type=int, default=50, help="channel embedding dimension (DyGFormer)")
    parser.add_argument("--max_input_sequence_length", type=int, default=32, help="input sequence length (DyGFormer)")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD", "RMSprop"], help="optimizer"
    )
    parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay")
    parser.add_argument("--patience", type=int, default=10, help="patience for early stopping")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.4, help="test split ratio")
    parser.add_argument("--num_runs", type=int, default=5, help="number of independent runs")
    parser.add_argument("--test_interval_epochs", type=int, default=10, help="epochs between test runs")
    parser.add_argument("--flow", type=str, default="False", choices=["True", "False"], help="whether to use flow-based scoring")
    try:
        args = parser.parse_args()
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    except:
        parser.print_help()
        sys.exit(0)

    return args
