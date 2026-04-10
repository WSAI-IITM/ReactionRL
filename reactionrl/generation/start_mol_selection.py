"""Starting molecule selection for COMA benchmark evaluation.

Selects diverse starting molecules for each target based on Tanimoto similarity,
scaffold similarity, and maximum common substructure (MCS).
"""
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS

from reactionrl.config import DATASETS_DIR


def _tanimoto_bulk(target_fp, candidate_fps):
    """Compute Tanimoto similarity between one fingerprint and many."""
    return np.array(DataStructs.BulkTanimotoSimilarity(target_fp, candidate_fps))


def select_start_mols(target_smi, candidate_smiles, n=10):
    """Select diverse starting molecules for a target molecule.

    Uses three criteria to select 3*n unique starting molecules:
    1. Top n by Tanimoto similarity to target
    2. Top n by scaffold Tanimoto similarity
    3. Top n by MCS atom count

    Args:
        target_smi: Target molecule SMILES.
        candidate_smiles: List of candidate starting molecule SMILES.
        n: Base count. Returns up to 3*n unique molecules.

    Returns:
        List of selected starting molecule SMILES.
    """
    target_mol = Chem.MolFromSmiles(target_smi)
    if target_mol is None:
        return []

    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)

    # Compute Tanimoto similarity to all candidates
    valid_indices = []
    candidate_fps = []
    for i, smi in enumerate(candidate_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_indices.append(i)
            candidate_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    if not valid_indices:
        return []

    sims = _tanimoto_bulk(target_fp, candidate_fps)
    valid_indices = np.array(valid_indices)

    selected = set()

    # Criterion 1: Top n by Tanimoto similarity
    top_sim = np.argsort(sims)[::-1][:n]
    selected.update(valid_indices[top_sim].tolist())

    # Criterion 2: Top n by scaffold similarity
    try:
        target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
        target_scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(target_scaffold, 2, nBits=2048)
        scaffold_fps = []
        scaffold_valid = []
        for j, i in enumerate(valid_indices):
            mol = Chem.MolFromSmiles(candidate_smiles[i])
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_fps.append(AllChem.GetMorganFingerprintAsBitVect(scaffold, 2, nBits=2048))
                scaffold_valid.append(j)
            except Exception:
                pass
        if scaffold_fps:
            scaffold_sims = _tanimoto_bulk(target_scaffold_fp, scaffold_fps)
            scaffold_valid = np.array(scaffold_valid)
            top_scaffold = np.argsort(scaffold_sims)[::-1][:n]
            selected.update(valid_indices[scaffold_valid[top_scaffold]].tolist())
    except Exception:
        pass

    # Criterion 3: Top n by MCS atom count
    try:
        mcs_scores = []
        mcs_valid = []
        # Only check top candidates by similarity to limit MCS computation
        check_indices = np.argsort(sims)[::-1][:min(3 * n, len(sims))]
        for j in check_indices:
            i = valid_indices[j]
            mol = Chem.MolFromSmiles(candidate_smiles[i])
            try:
                mcs = rdFMCS.FindMCS([target_mol, mol], timeout=1, completeRingsOnly=True)
                mcs_scores.append(mcs.numAtoms)
                mcs_valid.append(i)
            except Exception:
                pass
        if mcs_scores:
            mcs_scores = np.array(mcs_scores)
            mcs_valid = np.array(mcs_valid)
            top_mcs = np.argsort(mcs_scores)[::-1][:n]
            selected.update(mcs_valid[top_mcs].tolist())
    except Exception:
        pass

    return [candidate_smiles[i] for i in selected]


def load_start_mols(pickle_path=None):
    """Load starting molecules from pickle file.

    Args:
        pickle_path: Path to pickle file. Defaults to
            datasets/my_uspto/unique_start_mols.pickle.

    Returns:
        List of SMILES strings.
    """
    if pickle_path is None:
        pickle_path = str(DATASETS_DIR / "my_uspto/unique_start_mols.pickle")
    with open(pickle_path, 'rb') as f:
        mols = pickle.load(f)
    # Convert from pandas Series if needed
    if hasattr(mols, 'tolist'):
        mols = mols.tolist()
    return mols
