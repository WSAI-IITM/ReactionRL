import tqdm
import numpy as np
import pandas as pd
import functools
from rdkit import Chem
from multiprocessing import Pool, cpu_count

from reactionrl.actions import get_applicable_actions, apply_action


def _get_app_act_count(smile):
    act = get_applicable_actions(Chem.MolFromSmiles(smile))
    if len(act.shape) > 0:
        return act.shape[0]
    return 0


def calc_start_mol_prob_dist(start_mols, processes=5):
    """Computes sampling probability for start molecules based on applicable action counts."""
    print("Calculating probability for start mol sampling")
    applicable_action_count = []
    with Pool(processes) as p:
        for c in tqdm.tqdm(p.imap(_get_app_act_count, start_mols, chunksize=100), total=len(start_mols)):
            applicable_action_count.append(c)

    applicable_action_count = np.array(applicable_action_count)
    return applicable_action_count / applicable_action_count.sum()


def _generate_data(smile, steps):
    """Generates random trajectory from a starting molecule."""
    mol = Chem.MolFromSmiles(smile)

    df = pd.DataFrame(columns=['reactant', 'rsub', 'rcen', 'rsig', 'rsig_cs_indices', 'psub', 'pcen', 'psig', 'psig_cs_indices', 'product', 'step'])
    index = []

    # Get sequences
    try:
        for i in range(steps):
            actions = get_applicable_actions(mol)
            if actions.shape[0] == 0:
                raise Exception("No actions applicable.....")

            # Apply a random action
            rand_idx = np.random.randint(0, actions.shape[0])
            product = apply_action(mol, *actions.iloc[rand_idx])

            # Add it to df
            df.loc[df.shape[0], :] = [Chem.MolToSmiles(mol)] + actions.iloc[rand_idx].tolist() + [Chem.MolToSmiles(product), i]
            index.append(actions.iloc[rand_idx].name)

            # Next reactant = product
            mol = product
    except Exception as e:
        return pd.DataFrame(columns=['reactant', 'rsub', 'rcen', 'rsig', 'rsig_cs_indices', 'psub', 'pcen', 'psig', 'psig_cs_indices', 'product', 'step'])

    # Fix index
    df.index = index

    # Fix target
    df["product"] = Chem.MolToSmiles(product)

    # Fix steps
    df["step"] = df.shape[0] - df["step"]

    return df


def generate_data(N, steps, start_mols, start_mol_prob, processes=5):
    """Generate N trajectory samples of specified length.

    Args:
        N: Target number of (reactant, action, product) rows to generate.
        steps: Number of steps per trajectory.
        start_mols: List of starting molecule SMILES.
        start_mol_prob: Sampling probabilities for start_mols (from calc_start_mol_prob_dist).
        processes: Number of parallel workers. Use 1 for easier debugging.

    Returns:
        DataFrame with columns: reactant, rsub, rcen, rsig, rsig_cs_indices,
        psub, pcen, psig, psig_cs_indices, product, step.
    """
    df_list = []
    final_shape = 0
    smiles_per_random_sample = 1000

    print("Creating dataset...")
    if processes > 1:
        with Pool(processes) as p, tqdm.tqdm(total=N) as pbar:
            while final_shape < N:
                smiles = np.random.choice(start_mols, size=(smiles_per_random_sample,), p=start_mol_prob)

                for new_df in p.imap_unordered(functools.partial(_generate_data, steps=steps), smiles, chunksize=10):
                    df_list.append(new_df)
                    final_shape += new_df.shape[0]

                pbar.update(final_shape - pbar.n)
    else:
        # Single-process mode (useful for debugging)
        with tqdm.tqdm(total=N) as pbar:
            while final_shape < N:
                smiles = np.random.choice(start_mols, size=(smiles_per_random_sample,), p=start_mol_prob)
                for smi in smiles:
                    new_df = _generate_data(smi, steps=steps)
                    df_list.append(new_df)
                    final_shape += new_df.shape[0]
                pbar.update(final_shape - pbar.n)

    main_df = pd.concat(df_list)
    return main_df
