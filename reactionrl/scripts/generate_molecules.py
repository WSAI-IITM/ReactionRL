"""CLI for molecule generation via beam search (Algorithm 2).

Generates molecules similar to targets by iteratively applying chemical
transformations selected by a trained actor-critic model.

Usage:
    python -m reactionrl.scripts.generate_molecules \\
        --model-path output/supervised/actor-critic/.../model.pth \\
        --source-smiles "CCO" "CCCC" \\
        --target-smiles "CC(=O)O" "CCCCO" \\
        --steps 5 --topk-actor 50 --topk-critic 5 \\
        --output results/generation.pickle --cuda 0
"""
import argparse
import pickle
import torch

from reactionrl.generation.beam_search import prepare_action_data, generate_molecules


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate molecules via beam search (Algorithm 2)."
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model .pth file")
    group_src = parser.add_mutually_exclusive_group(required=True)
    group_src.add_argument("--source-smiles", nargs="+", type=str,
                           help="Source molecule SMILES")
    group_src.add_argument("--source-file", type=str,
                           help="File with source SMILES, one per line")
    group_tgt = parser.add_mutually_exclusive_group(required=True)
    group_tgt.add_argument("--target-smiles", nargs="+", type=str,
                           help="Target molecule SMILES")
    group_tgt.add_argument("--target-file", type=str,
                           help="File with target SMILES, one per line")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of generation steps (default: 5)")
    parser.add_argument("--topk-actor", type=int, default=50,
                        help="Actor pre-filter count B_A (default: 50)")
    parser.add_argument("--topk-critic", type=int, default=5,
                        help="Beam width B after critic re-ranking (default: 5)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Multiprocessing workers (default: 8)")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="GPU device index (-1 for CPU)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pickle file path")
    return parser.parse_args()


def _load_smiles(smiles_list, file_path):
    """Load SMILES from args or file."""
    if smiles_list is not None:
        return smiles_list
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = get_args()

    # Device
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()

    # Load SMILES
    source_smiles = _load_smiles(args.source_smiles, args.source_file)
    target_smiles = _load_smiles(args.target_smiles, args.target_file)
    assert len(source_smiles) == len(target_smiles), \
        f"Source ({len(source_smiles)}) and target ({len(target_smiles)}) counts must match"
    print(f"{len(source_smiles)} source-target pairs")

    # Prepare action data
    print("Loading action space...")
    _, action_rsigs, action_psigs = prepare_action_data()

    # Generate
    traj_dict, sim_dict = generate_molecules(
        model, source_smiles, target_smiles,
        action_rsigs, action_psigs, device,
        steps=args.steps,
        topk_actor=args.topk_actor,
        topk_critic=args.topk_critic,
        num_workers=args.num_workers,
    )

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump({"traj": traj_dict, "sim": sim_dict}, f)
    print(f"Saved {len(traj_dict)} molecules to {args.output}")

    # Summary
    sims = list(sim_dict.values())
    if sims:
        import numpy as np
        sims = np.array(sims)
        print(f"Similarity stats: mean={sims.mean():.4f}, max={sims.max():.4f}, "
              f">{0.4:.0%}: {(sims > 0.4).sum()}")


if __name__ == "__main__":
    main()
