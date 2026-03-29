"""
Generates and deduplicates action embeddings.
Replaces the old version that depended on ChemRL (online RL env).
Uses the GIN embedder directly instead.
"""
from rdkit import Chem
import numpy as np
import pandas as pd
import tqdm
import pickle
import torch

from reactionrl.embeddings.gin import Zinc_GIN_Embedder


# Load csv
csv_path = "datasets/my_uspto/action_dataset-filtered.csv"
dataset = pd.read_csv(csv_path, index_col=0)
print(dataset.shape)

# Load embedder
embedder = Zinc_GIN_Embedder()
action_embeddings = []
failures = []

# Get all embeddings
for i in tqdm.tqdm(range(dataset.shape[0])):
    row = dataset.iloc[i][["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]]
    try:
        rsig_emb = embedder.mol_to_embedding(row["rsig"])
        psig_emb = embedder.mol_to_embedding(row["psig"])
        embedding = torch.concatenate([rsig_emb, psig_emb]).numpy()
        action_embeddings.append(embedding)
    except Exception as e:
        # Try with explicit hydrogen
        try:
            rsig_mol = Chem.AddHs(Chem.MolFromSmiles(row["rsig"]))
            psig_mol = Chem.AddHs(Chem.MolFromSmiles(row["psig"]))
            rsig_emb = embedder.mol_to_embedding(rsig_mol)
            psig_emb = embedder.mol_to_embedding(psig_mol)
            embedding = torch.concatenate([rsig_emb, psig_emb]).numpy()
            action_embeddings.append(embedding)
            failures.append(i)
        except Exception as e2:
            print(f"Failed for row {i}: {e2}")
            action_embeddings.append(np.zeros_like(action_embeddings[-1]) if action_embeddings else np.zeros(600))
            failures.append(i)


# Create hash for embeddings for fast search
action_embedding_hash = list(map(lambda x: hash(" ".join(map(str, x))), action_embeddings))
action_embeddings = np.array(action_embeddings)
action_embedding_hash = np.array(action_embedding_hash)

# Remove hash collisions (some actions that weren't detected as different before)
x, y = np.unique(action_embedding_hash, return_counts=True)
collision_hashes = x[y != 1]
idx_arr = np.isin(action_embedding_hash, collision_hashes, invert=True)

for coll in collision_hashes:
    idx_arr[np.where(action_embedding_hash == coll)[0][0]] = True

assert idx_arr.sum() == action_embedding_hash.shape[0] - y[y != 1].sum() + y[y != 1].shape[0]


# Set embedding hash as index
dataset.index = action_embedding_hash

dataset = dataset.loc[idx_arr]

dataset.to_csv(csv_path)

# Dump the hash->embedding dict
hash_to_embedding_map = {action_embedding_hash[i]: action_embeddings[i] for i in range(len(action_embeddings))}
pickle.dump(hash_to_embedding_map, open("datasets/my_uspto/action_embeddings.pickle", 'wb'))
