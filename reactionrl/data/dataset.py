"""Offline RL dataset for action prediction training.

Loads trajectory data, computes applicable action indices, and prepares
torchdrug molecule batches for efficient training.
"""
import numpy as np
import pandas as pd
from torchdrug import data
from multiprocessing import Pool
from collections import namedtuple

from reactionrl.data.molecule_utils import molecule_from_smile
from reactionrl.actions.action_space import get_default_action_space
from reactionrl.training.embedding_helpers import get_emb_indices_and_correct_idx

OfflineRLSplit = namedtuple("OfflineRLSplit", [
    "idx", "reactants", "products", "rsigs", "psigs"
])


class OfflineRLDataset:
    """Dataset for offline RL training.

    Loads a CSV of (reactant, action, product) trajectories, computes
    applicable action indices for each sample, and packs molecules into
    torchdrug batches for efficient GPU training.

    Usage:
        dataset = OfflineRLDataset("datasets/offlineRL/5steps_train.csv")
        dataset.prepare(num_workers=8)
        train, valid = dataset.split(train_frac=0.8)

    Args:
        csv_path: Path to training CSV (output of generate_data).
        action_space: ActionSpace instance. Defaults to the global singleton.
        device: Torch device for molecule batches.
        random_state: Seed for shuffling. None for non-deterministic shuffle.
    """
    def __init__(self, csv_path, action_space=None, device="cpu", random_state=None):
        self.df = pd.read_csv(csv_path, index_col=0).sample(frac=1, random_state=random_state)
        self.device = device
        if action_space is None:
            action_space = get_default_action_space()
        self._action_space = action_space

        # Populated by prepare()
        self.correct_applicable_indices = []
        self.correct_action_dataset_indices = []
        self.action_embedding_indices = []
        self._action_rsigs = None
        self._action_psigs = None

    def prepare(self, num_workers=8):
        """Pack molecules into torchdrug batches and compute action indices.

        This is the expensive step — it runs get_applicable_actions() for every
        sample using multiprocessing.

        Args:
            num_workers: Number of parallel workers. Each worker lazily loads
                its own copy of the ActionSpace (~100MB memory each).
        """
        action_dataset = self._action_space.dataset[
            ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
        ]

        # Pack action signatures for fast embedding computation
        self._action_rsigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["rsig"])))
        self._action_psigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["psig"])))

        # Compute correct action indices for each training sample
        self.correct_applicable_indices = []
        self.correct_action_dataset_indices = []
        self.action_embedding_indices = []

        with Pool(num_workers) as p:
            for indices_used_for_data, correct_app_idx, correct_act_idx in p.imap(
                get_emb_indices_and_correct_idx, self.df.iterrows(), chunksize=50
            ):
                self.action_embedding_indices.append(indices_used_for_data)
                self.correct_applicable_indices.append(correct_app_idx)
                self.correct_action_dataset_indices.append(correct_act_idx)

    def split(self, train_frac=0.8):
        """Split into training and validation sets.

        Args:
            train_frac: Fraction of data for training (default 0.8).

        Returns:
            Tuple of (train_split, valid_split) as OfflineRLSplit namedtuples.
        """
        n = self.df.shape[0]
        train_idx = np.arange(0, int(n * train_frac))
        valid_idx = np.arange(int(n * train_frac), n)

        train_split = OfflineRLSplit(
            idx=train_idx,
            reactants=data.Molecule.pack(list(map(molecule_from_smile, self.df.iloc[train_idx]["reactant"]))).to(self.device),
            products=data.Molecule.pack(list(map(molecule_from_smile, self.df.iloc[train_idx]["product"]))).to(self.device),
            rsigs=data.Molecule.pack(list(map(molecule_from_smile, self.df.iloc[train_idx]["rsig"]))).to(self.device),
            psigs=data.Molecule.pack(list(map(molecule_from_smile, self.df.iloc[train_idx]["psig"]))).to(self.device),
        )
        valid_split = OfflineRLSplit(
            idx=valid_idx,
            reactants=data.Molecule.pack(list(map(molecule_from_smile, self.df.iloc[valid_idx]["reactant"]))).to(self.device),
            products=data.Molecule.pack(list(map(molecule_from_smile, self.df.iloc[valid_idx]["product"]))).to(self.device),
            rsigs=data.Molecule.pack(list(map(molecule_from_smile, self.df.iloc[valid_idx]["rsig"]))).to(self.device),
            psigs=data.Molecule.pack(list(map(molecule_from_smile, self.df.iloc[valid_idx]["psig"]))).to(self.device),
        )
        return train_split, valid_split

    @property
    def action_rsigs(self):
        return self._action_rsigs

    @property
    def action_psigs(self):
        return self._action_psigs
