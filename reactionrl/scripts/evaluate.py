"""CLI for evaluating a trained model on COMA benchmarks.

Runs the full evaluation pipeline: load test molecules, select starting molecules,
generate candidates via beam search, and compute COMA metrics.

Usage:
    python -m reactionrl.scripts.evaluate \\
        --model-path output/supervised/actor-critic/.../model.pth \\
        --property qed --cuda 0

COMA test data must be placed at datasets/coma/{property}/rdkit_test.txt.
Download from: https://github.com/wengong-jin/iclr19-graph2graph
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from reactionrl.config import DATASETS_DIR, OUTPUT_DIR
from reactionrl.generation.beam_search import prepare_action_data, generate_molecules
from reactionrl.generation.start_mol_selection import select_start_mols, load_start_mols
from reactionrl.evaluation.evaluate import evaluate_metric
from reactionrl.evaluation.properties import (
    qed as eval_qed,
    penalized_logp as eval_penalized_logp,
    similarity as eval_similarity,
)
from reactionrl.rewards.properties import drd2 as eval_drd2

# Property scoring functions keyed by benchmark name
SCORING_FNS = {
    "qed": lambda smi: eval_qed(smi) if smi else 0.0,
    "drd2": lambda smi: eval_drd2(Chem.MolFromSmiles(smi)) if smi else 0.0,
    "logp04": lambda smi: eval_penalized_logp(smi) if smi else 0.0,
    "logp06": lambda smi: eval_penalized_logp(smi) if smi else 0.0,
}

# (x, y) coefficient combos used in the paper for each benchmark
COEFF_COMBOS = {
    "qed": [(0, 1), (1, 0), (1, 1)],
    "drd2": [(0, 1), (1, 0), (1, 1)],
    "logp04": [(0, 1), (1, 0), (25, 1)],
    "logp06": [(0, 1), (1, 0), (25, 1)],
}


def check_coma_data(property_name):
    """Check if COMA test data exists and return path or print instructions."""
    test_path = DATASETS_DIR / f"coma/{property_name}/rdkit_test.txt"
    if not test_path.exists():
        print(f"COMA test data not found at: {test_path}")
        print()
        print("To download the benchmark test data:")
        print("  1. Clone: git clone https://github.com/wengong-jin/iclr19-graph2graph")
        print(f"  2. Copy the test file to: {test_path}")
        print()
        print("The test files are typically found under data/{property}/test.txt")
        print("in the iclr19-graph2graph repository.")
        return None
    return str(test_path)


def load_test_targets(test_path):
    """Load target SMILES from a COMA test file (one SMILES per line)."""
    targets = []
    with open(test_path) as f:
        for line in f:
            smi = line.strip().split()[0]  # Handle lines with extra columns
            if smi:
                targets.append(smi)
    return targets


def select_top_molecules(trajectory_dict, similarity_dict, target_smiles,
                         scoring_fn, num_decode=20, x=1, y=1):
    """Select top molecules per target by x*similarity + y*property.

    Args:
        trajectory_dict: Maps composite keys to product SMILES.
        similarity_dict: Maps same keys to Tanimoto similarity.
        target_smiles: List of target SMILES (indexed by first element of key).
        scoring_fn: Property scoring function (SMILES -> float).
        num_decode: Number of molecules to select per target.
        x: Similarity coefficient.
        y: Property coefficient.

    Returns:
        DataFrame with columns [source, target, similarity, property_target, property_source]
        grouped in blocks of num_decode rows per target.
    """
    # Group molecules by target index
    target_molecules = {}
    for key, smi in trajectory_dict.items():
        ti = int(key.split("_")[0])
        if ti not in target_molecules:
            target_molecules[ti] = []
        sim = similarity_dict.get(key, 0.0)
        target_molecules[ti].append((smi, sim, key))

    n_targets = len(target_smiles)
    rows = []

    for ti in range(n_targets):
        target_smi = target_smiles[ti]
        candidates = target_molecules.get(ti, [])

        if not candidates:
            # Pad with dummy entries
            for _ in range(num_decode):
                rows.append([target_smi, target_smi, 0.0, 0.0, 0.0])
            continue

        # Score each candidate
        scored = []
        for smi, sim, key in candidates:
            prop = scoring_fn(smi)
            combined = x * sim + y * prop
            scored.append((smi, sim, prop, combined))

        # Sort by combined score, take top num_decode
        scored.sort(key=lambda t: t[3], reverse=True)
        selected = scored[:num_decode]

        # Compute source property
        source_prop = scoring_fn(target_smi)

        for smi, sim, prop, _ in selected:
            rows.append([target_smi, smi, sim, prop, source_prop])

        # Pad if fewer than num_decode
        for _ in range(num_decode - len(selected)):
            rows.append([target_smi, target_smi, 0.0, 0.0, source_prop])

    return pd.DataFrame(rows, columns=["source", "target", "similarity",
                                        "property_target", "property_source"])


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on COMA benchmarks."
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model .pth file")
    parser.add_argument("--property", type=str, required=True,
                        choices=["qed", "drd2", "logp04", "logp06"],
                        help="Benchmark property to evaluate")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="GPU device index (-1 for CPU)")
    parser.add_argument("--num-start-mols", type=int, default=10,
                        help="Base count for starting mols per target (selects 3x this)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Generation steps (default: 5)")
    parser.add_argument("--topk-actor", type=int, default=50,
                        help="Actor pre-filter B_A (default: 50)")
    parser.add_argument("--topk-critic", type=int, default=5,
                        help="Critic beam width B (default: 5)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Multiprocessing workers (default: 8)")
    parser.add_argument("--num-decode", type=int, default=20,
                        help="Top molecules per target for metrics (default: 20)")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Limit number of test targets (for quick testing)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output/coma_eval/{property})")
    return parser.parse_args()


def main():
    args = get_args()

    # Check data
    test_path = check_coma_data(args.property)
    if test_path is None:
        return

    # Device
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # Output dir
    output_dir = args.output_dir or str(OUTPUT_DIR / f"coma_eval/{args.property}")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()

    # Load action data
    print("Loading action space...")
    _, action_rsigs, action_psigs = prepare_action_data()

    # Load start molecules and test targets
    start_mols = load_start_mols()
    test_targets = load_test_targets(test_path)
    if args.max_targets is not None:
        test_targets = test_targets[:args.max_targets]
    print(f"Evaluating on {len(test_targets)} test targets")

    scoring_fn = SCORING_FNS[args.property]

    # Generate molecules for all targets
    all_traj = {}
    all_sim = {}
    target_offset = 0

    for t_idx, target_smi in enumerate(test_targets):
        print(f"\nTarget {t_idx + 1}/{len(test_targets)}: {target_smi}")

        # Select starting molecules
        selected_starts = select_start_mols(target_smi, start_mols, n=args.num_start_mols)
        if not selected_starts:
            print("  No valid starting molecules found, skipping")
            continue

        # Build source-target pairs (all sources paired with this target)
        source_list = selected_starts
        target_list = [target_smi] * len(source_list)

        # Generate
        traj, sim = generate_molecules(
            model, source_list, target_list,
            action_rsigs, action_psigs, device,
            steps=args.steps,
            topk_actor=args.topk_actor,
            topk_critic=args.topk_critic,
            num_workers=args.num_workers,
        )

        # Remap keys to use global target index
        for key, smi in traj.items():
            # Replace the source index prefix with global target index
            parts = key.split("_")
            global_key = f"{target_offset}_{('_'.join(parts[1:]))}" if len(parts) > 1 else str(target_offset)
            all_traj[global_key] = smi
            all_sim[global_key] = sim[key]

        target_offset += 1

        # Save per-target results
        with open(os.path.join(output_dir, f"target_{t_idx}.pickle"), 'wb') as f:
            pickle.dump({"traj": traj, "sim": sim, "target": target_smi}, f)

    if target_offset == 0:
        print("No targets processed.")
        return

    # Evaluate with different (x, y) coefficient combinations
    print(f"\n{'=' * 60}")
    print(f"Results for {args.property} benchmark")
    print(f"{'=' * 60}")

    actual_targets = test_targets[:target_offset]
    for x, y in COEFF_COMBOS[args.property]:
        print(f"\n--- Coefficients: x={x}, y={y} ---")
        df = select_top_molecules(
            all_traj, all_sim, actual_targets,
            scoring_fn, num_decode=args.num_decode, x=x, y=y,
        )

        results = evaluate_metric(
            df, smiles_train_high=[],
            num_decode=args.num_decode,
            threshold_pro=0.0, threshold_improve=0.0,
            list_threshold_sim=[0.4],
        )

        print("Metrics:")
        print(results["metrics"].to_string())
        print("Success rates:")
        print(results["success_rate"].to_string())

        # Save
        result_file = os.path.join(output_dir, f"results_x{x}_y{y}.pickle")
        with open(result_file, 'wb') as f:
            pickle.dump({"df": df, "results": results, "x": x, "y": y}, f)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
