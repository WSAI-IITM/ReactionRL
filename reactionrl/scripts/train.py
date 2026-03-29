"""CLI entry point for offline RL training.

Usage:
    python -m reactionrl.scripts.train --steps 5 --model-type actor-critic --actor-loss PG --cuda 0
"""
import argparse
import numpy as np
import torch

from reactionrl.config import TrainingConfig, DATASETS_DIR
from reactionrl.models import MODEL_REGISTRY
from reactionrl.data.dataset import OfflineRLDataset
from reactionrl.training.trainer import OfflineRLTrainer


def get_args():
    parser = argparse.ArgumentParser(description="Train an offline RL model for action prediction.")
    parser.add_argument("--steps", type=int, required=True, help="Trajectory length (matches dataset filename)")
    parser.add_argument("--model-type", type=str, choices=["actor", "critic", "actor-critic"], required=True, help="Type of model to train")
    parser.add_argument("--actor-loss", type=str, choices=["mse", "PG"], default="PG", help="Actor loss type (default: PG)")
    parser.add_argument("--negative-selection", type=str, choices=["random", "closest", "e-greedy", "combined"], default="combined", help="Negative selection strategy")
    parser.add_argument("--cuda", type=int, default=-1, help="GPU device index (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--num-workers", type=int, default=None, help="Multiprocessing workers for data prep")
    return parser.parse_args()


def main():
    args = get_args()

    # Device
    if args.cuda >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    print(f"Using device {device}")

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Config
    config = TrainingConfig(
        steps=args.steps,
        model_type=args.model_type,
        actor_loss=args.actor_loss,
        negative_selection=args.negative_selection,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
    )

    # Load data
    csv_path = str(DATASETS_DIR / f"offlineRL/{args.steps}steps_train.csv")
    dataset = OfflineRLDataset(csv_path, device=device)
    num_workers = args.num_workers if args.num_workers is not None else config.num_workers
    dataset.prepare(num_workers=num_workers)

    # Init model - pass architecture params from config
    model_cls = MODEL_REGISTRY[args.model_type]
    model_kwargs = {"gin_model_path": config.get_gin_model_path(), "hidden_size": config.hidden_size}
    if args.model_type == "actor":
        model_kwargs["num_hidden"] = config.actor_num_hidden
    elif args.model_type == "critic":
        model_kwargs["num_hidden"] = config.critic_num_hidden
    elif args.model_type == "actor-critic":
        model_kwargs["actor_num_hidden"] = config.actor_num_hidden
        model_kwargs["critic_num_hidden"] = config.critic_num_hidden
    model = model_cls(**model_kwargs).to(torch.device(device))

    # Split data
    train_split, valid_split = dataset.split(train_frac=config.train_frac)

    print("Train and valid data shapes:")
    print(train_split.reactants.batch_size, train_split.products.batch_size,
          valid_split.reactants.batch_size, valid_split.products.batch_size)

    # Train
    trainer = OfflineRLTrainer(model, dataset, config)
    trainer.train(train_split, valid_split)
    trainer.save()


if __name__ == "__main__":
    main()
