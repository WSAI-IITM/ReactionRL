import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFMCS
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import rdkit.Chem.QED as QED

import reactionrl.evaluation.sascorer as sascorer


def get_kekuleSmiles(smi):
    mol = Chem.MolFromSmiles(smi)
    smi_rdkit = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=False,
                    kekuleSmiles=True,
                    rootedAtAtom=-1,
                    canonical=True,
                    allBondsExplicit=False,
                    allHsExplicit=False
                )
    return smi_rdkit


def rdkit_kekulize_handling(original_fn):
    def wrapper_fn(*args, **kwargs):
        try:
            score = original_fn(*args, **kwargs)
        except Chem.rdchem.KekulizeException:
            score = 0.
        finally:
            return score
    return wrapper_fn


@rdkit_kekulize_handling
def qed(s):
    if s is None:
        return 0.
    else:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return 0.
            else:
                return QED.qed(mol)
        except:
            return 0.


# Modified from https://github.com/bowenliu16/rl_graph_generation
def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    ## logp
    log_p = Descriptors.MolLogP(mol)
    ## synthetic accessiblity
    SA = -sascorer.calculateScore(mol)
    ## cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    cycle_length = max([len(j) for j in cycle_list]) if len(cycle_list) > 0 else 0
    cycle_score = min(0, 6 - cycle_length)

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


def similarity(a, b):
    if a is None or b is None:
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    else:
        fp1 = GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        fp2 = GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
        return TanimotoSimilarity(fp1, fp2)


def mcs_similarity(a, b):
    if a is None or b is None:
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    else:
        mcs = rdFMCS.FindMCS([amol, bmol], completeRingsOnly=True)
        sim_atom = 2 * mcs.numAtoms / (amol.GetNumAtoms() + bmol.GetNumAtoms())
        sim_bond = 2 * mcs.numBonds / (amol.GetNumBonds() + bmol.GetNumBonds())
        return 0.5 * (sim_atom + sim_bond)


class FastTanimotoOneToBulk:
    def __init__(self, bs):
        self.bs = bs
        self.b_fps = np.vstack([self._fingerprints_from_smi(smi) for smi in self.bs])

    def __call__(self, a):
        a_fp = self._fingerprints_from_smi(a)
        return (a_fp & self.b_fps).sum(axis=1) / (a_fp | self.b_fps).sum(axis=1)

    def _fingerprints_from_smi(self, smi):
        mol = Chem.MolFromSmiles(smi)
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
        nfp = np.array([b == '1' for b in fp.ToBitString()])
        return nfp
