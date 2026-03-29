'''
This code reads action_dataset-filtered.csv and extracts all unique mols.
'''

import pandas as pd
from rdkit import Chem
from reactionrl.utils.mol_utils import get_mol_certificate
import tqdm

df = pd.read_csv("datasets/my_uspto/action_dataset-filtered.csv")
df = df[df["reactant_works"]]
mols = df["reactants"]
print(df.shape, mols.shape)
# certi to mol(df index) dict
d = {}

for i, mol in tqdm.tqdm(enumerate(mols)):
    certi = get_mol_certificate(Chem.MolFromSmiles(mol))
    if certi not in d:
        d[certi] = [i]
    else:
        d[certi].append(i)

# Certificates aren't unique
unique_idx_list = []

def get_unique_mol_idx(idx_list):
    idx_list = list(idx_list)
    unique_idx = [idx_list.pop(0)]
    unique_mols = [Chem.MolFromSmiles(df.iloc[unique_idx[0]]["reactants"])]

    # Func to check if 2 mols are same
    def same(mol1, mol2):
        if mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1):
            return True
        return False

    # Get unique mols from the list
    for idx in idx_list:
        mol = Chem.MolFromSmiles(df.iloc[idx]["reactants"])
        if not any(list(map(lambda x: same(mol, x), unique_mols))):
            unique_idx.append(idx)
            unique_mols.append(mol)
    return unique_idx

for key in tqdm.tqdm(d):
    unique_idx_list.extend(get_unique_mol_idx(d[key]))

print("Unique mols:", len(unique_idx_list))

# dump mols
mols.iloc[unique_idx_list].to_pickle("datasets/my_uspto/unique_start_mols.pickle")
