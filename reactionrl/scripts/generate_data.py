import argparse
import os
import pickle
import numpy as np
from multiprocessing import cpu_count

from reactionrl.config import DATASETS_DIR
from reactionrl.data.generation import calc_start_mol_prob_dist, generate_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=100000, help="Number of data points to use")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps per trajectory")
    parser.add_argument("--processes", type=int, default=int(0.8 * cpu_count()), help="Number of CPU cores to use for multiprocessing")
    return parser.parse_args()


def main():
    args = get_args()

    with open(str(DATASETS_DIR / "my_uspto/unique_start_mols.pickle"), 'rb') as f:
        start_mols = pickle.load(f)
    start_mol_prob = calc_start_mol_prob_dist(start_mols, processes=args.processes)

    # Dump train and test (test is 20% of train samples)
    os.makedirs(str(DATASETS_DIR / "offlineRL"), exist_ok=True)
    for data_type in ["train", "test"]:
        file = str(DATASETS_DIR / f"offlineRL/{args.steps}steps_{data_type}.csv")
        samples = args.train_samples if data_type == "train" else int(0.2 * args.train_samples)
        df = generate_data(samples, args.steps, start_mols, start_mol_prob, processes=args.processes)
        df.to_csv(file)
        print(f"Dumped at {file}. Shape = {df.shape}")


if __name__ == "__main__":
    main()
