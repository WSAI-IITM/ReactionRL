"""Second-pass action filter: verifies apply_action produces the expected product.

Marks actions as invalid if apply_action(reactant, action) does not match
the recorded product SMILES.
"""
from rdkit import Chem
import numpy as np
import pandas as pd
import tqdm

from reactionrl.actions import apply_action

dataset = pd.read_csv("datasets/my_uspto/action_dataset-filtered.csv", index_col=0)

mts = Chem.MolToSmiles
mfs = Chem.MolFromSmiles

def molecule_equality(m1, m2):
    if isinstance(m1, str):
        m1 = mfs(m1)
    if isinstance(m2, str):
        m2 = mfs(m2)
    m1 = mfs(mts(m1))
    m2 = mfs(mts(m2))

    if m1 is None or m2 is None:
        return False
    return (mts(m1, 1) == mts(m2, 1)) or (m1.HasSubstructMatch(m2) and m2.HasSubstructMatch(m1))

l = np.array([True] * dataset.shape[0])

for i in tqdm.tqdm(range(dataset.shape[0])):
    row = dataset.iloc[i]
    reactant = mfs(row["reactants"])
    action = row[["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]]
    product = mfs(row["products"])

    try:
        p = apply_action(reactant, *action)
    except:
        continue

    if not molecule_equality(product, p):
        l[i] = False

dataset["action_works"] = (dataset["action_works"].values & l)
dataset.to_csv("datasets/my_uspto/action_dataset-filtered.csv")
