import argparse
import os
from argparse import Namespace
from datetime import datetime

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import Model
from helper import str2bool

from wandb_helper import WandbCallback

def main():
    args = _parse_args()
    pl.seed_everything(args.seed)
    model = _create_model(args)
    datamodule = _create_datamodule(args)
    logger = _create_logger(args)
    trainer = _create_trainer(args, logger)

    trainer.fit(model, datamodule)


def _create_trainer(args: Namespace, pl_logger) -> pl.Trainer:
    if args.wandb_logging:
        wandb_callbacks = [
            WandbCallback(args.gpus * args.batch_size),
        ]
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=wandb_callbacks if args.wandb_logging else None,
        logger=pl_logger,
    )
    return trainer


def _create_logger(args: Namespace):
    if args.wandb_logging:
        # Set up wandb logging
        pl_logger = WandbLogger(
            project=args.project,
            name=args.name,
            offline=False,
            log_model=False,
        )
        wandb.require(experiment="service")
    else:
        pl_logger = None

    return pl_logger


def _create_model(args: Namespace) -> Model:
    kwargs = {
        "learning_rate": args.learning_rate,
        "transformer_layers": args.transformer_layers,
        "use_transformer": args.use_transformer,
        "num_timesteps": args.num_timesteps,
        "p_loss": args.p_loss,
        "p_weight": args.p_weight,
        "p_reduction": args.p_reduction,
        "p_anneal": args.p_anneal,
        "activation": args.activation,
        "l1o_weight": args.l1o_weight,
        "l1f_weight": args.l1f_weight,
        "map_size": args.map_size,
        "anneal_lr": args.anneal_lr,
        "weight_decay": args.weight_decay,
        "full_decode": args.full_decode,
        "K_down": args.K_down,
        "xy_resolution": 64,
    }
    # Load checkpoint if specified
    if args.ckpt_file:
        model = Model.load_from_checkpoint(args.ckpt_file, **kwargs)
    else:
        model = Model(**kwargs)
    return model


def _create_datamodule(args: Namespace) -> DataModule:
    if args.val_batches == -1:
        val_dataset_size = None
    else:
        val_dataset_size = args.val_batches * args.batch_size

    datamodule = DataModule(
        n_gpus=args.gpus,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        val_dataset_size=val_dataset_size,
        train_dataset_size=args.train_batches,
        num_train_workers=0,
        num_timesteps=args.num_timesteps,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
    )
    datamodule.setup()
    return datamodule


def _parse_args() -> Namespace:
    wandb_run_name = f"{datetime.utcnow().isoformat(timespec='seconds')}"

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--wandb_logging",
        type=str2bool,
        default=True,
        help="whether to use wandb logging or not",
    )
    parser.add_argument(
        "--name", type=str, default=wandb_run_name, help="wandb run name (optional)"
    )
    parser.add_argument("--project", type=str, default="Model", help="wandb project name")
    parser.add_argument("--dataset_path", type=str, default='/data/asp', help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=2,
        help="memory usage when validating is higher than training; probably need to tune this to your GPU",
    )
    parser.add_argument("--learning_rate", type=float, default=20e-5)
    parser.add_argument(
        "--anneal_lr",
        type=str2bool,
        default=False,
        help="Anneal lr each epoch by 0.95**epoch",
    )
    parser.add_argument(
        "--ckpt_runid", type=str, default=None, help="wandb run to load checkpoint from"
    )
    parser.add_argument(
        "--ckpt_file", type=str, default=None, help="wandb checkpoint filename"
    )
    parser.add_argument(
        "--transformer_layers",
        type=int,
        default=4,
        help="number of layers per transformer stack",
    )
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument(
        "--val_batches",
        type=int,
        default=25,
        help="number of batches to run for validation",
    )
    parser.add_argument(
        "--train_batches",
        type=int,
        default=None,
        help="number of batches to run for training",
    )
    parser.add_argument(
        "--dataset", type=str, default="asp_surround", help="which dataset to train on"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--full_decode",
        type=str2bool,
        default=False,
        help="whether to decode the full image during training",
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="leaky",
        help="which activation function to use",
    )
    parser.add_argument(
        "--use_transformer",
        type=str2bool,
        default=True,
        help="Use transformer or linear layer",
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=6, help="number of timesteps to use"
    )
    parser.add_argument("--p_loss", type=float, default=2, help="which p loss to use")
    parser.add_argument("--p_weight", type=float, default=1, help="p loss weight")
    parser.add_argument("--l1o_weight", type=float, default=0, help="l1o loss weight")
    parser.add_argument("--l1f_weight", type=float, default=0, help="l1f loss weight")
    parser.add_argument(
        "--p_reduction", type=str, default="sum", help="which p reduction to use"
    )
    parser.add_argument("--p_anneal", type=float, default=0, help="Anneal p loss")
    parser.add_argument("--map_size", type=int, default=32, help="number of maps")
    parser.add_argument("--K_down", type=int, default=8, help="number of objects")

    return parser.parse_args()


if __name__ == "__main__":
    main()