"""Action space for molecular transformations.

Encapsulates the action dataset (reaction signature clusters) and provides
methods to find applicable actions for a molecule and apply them.

The ActionSpace loads a CSV of reaction templates and builds lookup indices
for fast substructure matching. For multiprocessing compatibility, use the
module-level convenience functions (get_applicable_actions, apply_action)
which delegate to a lazily-initialized singleton.
"""
from rdkit import Chem
import numpy as np
import pandas as pd
import tqdm
import pickle
import networkx as nx
import os
from filehash import FileHash

from reactionrl.utils.mol_utils import (
    get_mol_certificate,
    clean_hydrogen_in_smiles,
    mol_with_atom_index,
    mol_without_atom_index,
    find_connecting_atoms_not_in_sig,
    GetAtomWithAtomMapNum,
)
from reactionrl.config import MAIN_DIR


class ActionSpace:
    """Manages the set of available molecular transformation actions.

    Loads the action dataset CSV, builds reaction signature cluster indices,
    and provides methods to query applicable actions for any molecule and
    apply them to produce new molecules.

    Args:
        dataset_path: Path to action_dataset-filtered.csv. Defaults to
            datasets/my_uspto/action_dataset-filtered.csv under MAIN_DIR.
        main_dir: Root directory for dataset files. Defaults to config.MAIN_DIR.
    """
    def __init__(self, dataset_path=None, main_dir=None):
        if main_dir is None:
            main_dir = MAIN_DIR

        if dataset_path is None:
            dataset_path = os.path.join(main_dir, "datasets/my_uspto/action_dataset-filtered.csv")

        self.action_dataset_csv_path = dataset_path
        self.action_dataset_hash_path = dataset_path.replace(".csv", ".hash")
        self.dataset = pd.read_csv(self.action_dataset_csv_path, index_col=0)
        self.dataset = self.dataset[self.dataset["action_works"] & self.dataset["action_tested"]]

        path_to_rsig_cluster_dict = os.path.join(main_dir, "datasets/my_uspto/rsig_cluster_dict.pickle")
        path_to_certi_dict = os.path.join(main_dir, "datasets/my_uspto/certi_dict.pickle")

        # Get rsig cluster and certificate dictionaries
        try:
            action_dataset_csv_hash = FileHash("md5").hash_file(self.action_dataset_csv_path)
            if open(self.action_dataset_hash_path, 'r').readline() != action_dataset_csv_hash:
                print(f"{self.action_dataset_csv_path} file updated!")
                raise Exception(f"{self.action_dataset_csv_path} file updated!")
            self.rsig_cluster_to_rsig_d = pickle.load(open(path_to_rsig_cluster_dict, 'rb'))
            self.certificate_to_cluster_id_dict = pickle.load(open(path_to_certi_dict, 'rb'))
        except Exception as e:
            # Fetch the rsigs using the clusters
            print("Calculating rsig_cluster dict....")
            self.rsig_cluster_to_rsig_d = {}
            for cluster_id in tqdm.tqdm(self.dataset["rsig_clusters"].unique()):
                cluster_df = self.dataset[self.dataset["rsig_clusters"] == cluster_id]
                rsig = Chem.MolFromSmiles(cluster_df.iloc[0]["rsig"])
                self.rsig_cluster_to_rsig_d[cluster_id] = rsig
            pickle.dump(self.rsig_cluster_to_rsig_d, open(path_to_rsig_cluster_dict, 'wb'))

            # Make a mapping of certificates and cluster_ids
            print("Calculating certificate dict....")
            self.certificate_to_cluster_id_dict = {}
            for _id in tqdm.tqdm(self.rsig_cluster_to_rsig_d):
                C = get_mol_certificate(self.rsig_cluster_to_rsig_d[_id])
                if C in self.certificate_to_cluster_id_dict:
                    self.certificate_to_cluster_id_dict[C].append(_id)
                else:
                    self.certificate_to_cluster_id_dict[C] = [_id]
            pickle.dump(self.certificate_to_cluster_id_dict, open(path_to_certi_dict, 'wb'))

            print("Updating hash...")
            open(self.action_dataset_hash_path, 'w').write(FileHash("md5").hash_file(self.action_dataset_csv_path))

    def _add_immediate_neighbors(self, mol, indices, add_aromatic_cycles=True):
        '''
        If add_aromatic_cycles is true, adds the whole aromatic cycle part of any new atom added.
        Also, if add_aromatic_cycles is true, returns (indices, added_aromatic_cycle(bool))
        '''
        def _add_neighbors(idx_list):
            atoms = list(map(lambda x: mol.GetAtomWithIdx(int(x)), idx_list))
            neighbors = []
            for atom in atoms:
                neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
            return np.unique(neighbors).tolist()

        # first add immediate neighbors
        new_indices = _add_neighbors(indices)

        if set(new_indices) == set(indices):  # indices = whole molecule
            if add_aromatic_cycles:
                return indices, False
            return indices

        # if added neighbor is aromatic, we have to check for more neighbors
        added_aromatic_cycle = False
        if add_aromatic_cycles:
            if any(list(map(lambda idx: mol.GetAtomWithIdx(idx).GetIsAromatic(), list(set(new_indices) - set(indices))))):
                indices = list(new_indices)
                # if any aromatic atoms in neighbors, add them as well
                repeat = True
                while repeat:
                    repeat = False
                    for n in set(_add_neighbors(indices)) - set(indices):
                        if mol.GetAtomWithIdx(int(n)).GetIsAromatic():
                            indices.append(n)
                            repeat = True
                            added_aromatic_cycle = True
            return np.unique(indices), added_aromatic_cycle
        else:
            indices = new_indices

        return np.unique(indices)

    def _verify_action_applicability(self, mol, r_indices, cluster_id):
        mol = Chem.Mol(mol)
        rsig = Chem.MolFromSmiles(self.dataset[self.dataset["rsig_clusters"] == cluster_id].iloc[0]["rsig"])
        rsub = Chem.MolFromSmiles(self.dataset[self.dataset["rsig_clusters"] == cluster_id].iloc[0]["rsub"])
        rcen = self.dataset[self.dataset["rsig_clusters"] == cluster_id].iloc[0]["rcen"]
        rbond = self.dataset[self.dataset["rsig_clusters"] == cluster_id].iloc[0]["rbond"]
        rbond = list(map(float, rbond.replace("[", "").replace("]", "").replace(" ", "").split(",")))

        # Get the correct rsig_match
        rsig_matches = mol.GetSubstructMatches(rsig)
        if not rsig_matches:
            rsig_match = ()
        else:
            for rsig_match in rsig_matches:
                if not (set(rsig_match) - set(r_indices)):
                    break

        atm_map_nums = []
        for i in range(rsub.GetNumAtoms()):
            atm_map_nums.append(rsub.GetAtomWithIdx(i).GetAtomMapNum())

        try:
            rsub_match = np.array(rsig_match)[atm_map_nums].tolist()
        except Exception as e:
            return False

        # get neighbors
        atoms = list(map(lambda x: mol.GetAtomWithIdx(x), rsub_match))
        neighbors = []
        for atom in atoms:
            neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
        neighbors = np.unique(neighbors)

        if not set(neighbors) - set(rsig_match):
            return True
        return False

    def _get_mol_from_index_list(self, mol, indices):
        rw = Chem.RWMol(mol)
        rw.BeginBatchEdit()
        for idx in set(list(range(mol.GetNumAtoms()))) - set(indices):
            rw.RemoveAtom(idx)
        rw.CommitBatchEdit()
        return Chem.Mol(rw)

    def get_applicable_rsig_clusters(self, in_mol):
        # For each cut vertex, we find two disconnected components and search the smaller one in our index
        G = nx.from_numpy_array(Chem.GetAdjacencyMatrix(in_mol))
        applicable_clusters = []

        for x in nx.articulation_points(G):
            # Remove atom (not directly, otherwise the index resets)
            # First remove bonds to x
            in_mol_kekulized = Chem.Mol(in_mol)
            Chem.Kekulize(in_mol_kekulized, clearAromaticFlags=True)
            mw = Chem.RWMol(in_mol_kekulized)
            for n in mw.GetAtomWithIdx(x).GetNeighbors():
                mw.RemoveBond(x, n.GetIdx())

            # Find fragments
            mol_frags = list(Chem.rdmolops.GetMolFrags(mw))

            # Remove x from fragments
            mol_frags.remove((x,))

            # For each fragment except the biggest, add x and extract sub-molecule and search
            for frag in sorted(mol_frags, key=lambda x: len(x))[:-1]:
                indices = [x] + list(frag)
                aromatic_cycle_added = False

                for _ in range(2):
                    if aromatic_cycle_added:
                        continue
                    # we add neighbors twice to rsub and then search for rsig
                    indices, aromatic_cycle_added = self._add_immediate_neighbors(in_mol, indices)

                    candidate = self._get_mol_from_index_list(in_mol_kekulized, indices)
                    try:
                        Chem.SanitizeMol(candidate)
                    except Exception as e:
                        print(e)

                    # get certificate and search in rsig
                    cand_certi = get_mol_certificate(candidate)

                    if cand_certi in self.certificate_to_cluster_id_dict:
                        # Verify rsig
                        for cluster_id in self.certificate_to_cluster_id_dict[cand_certi]:
                            if cluster_id not in applicable_clusters:
                                if self._verify_action_applicability(in_mol, indices, cluster_id):
                                    applicable_clusters.append(cluster_id)
        return applicable_clusters

    def _mark_action_invalid(self, idx):
        # Load df
        temp_df = pd.read_csv(self.action_dataset_csv_path, index_col=0)

        # Mark action invalid
        temp_df.loc[idx, "action_works"] = False

        # Dump with random suffix to avoid race conditions
        num = np.random.random()
        temp_df.to_csv(self.action_dataset_csv_path + str(num))
        os.rename(self.action_dataset_csv_path + str(num), self.action_dataset_csv_path)

    def get_applicable_actions(self, mol, random_state=None):
        applicable_clusters = self.get_applicable_rsig_clusters(mol)
        return_format = ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
        return self.dataset[self.dataset["rsig_clusters"].isin(applicable_clusters)][return_format]

    def get_random_action(self, mol, random_state=None):
        applicable_clusters = self.get_applicable_rsig_clusters(mol)
        return_format = ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]

        # random sample
        sample = self.dataset[self.dataset["rsig_clusters"].isin(applicable_clusters)].sample(random_state=random_state)[return_format].iloc[0]
        return sample.values

    def _filter_sensible_rsig_matches(self, mol, rsig_matches, rsig, rsub, rcen):
        '''
        Checks if the rsub in given rsig only has neighbors within rsig
        '''
        # Get the atoms in rsig corresponding to rsub
        rsub_atom_indices = []
        for atom in rsub.GetAtoms():
            rsub_atom_indices.append(atom.GetAtomMapNum())
        rsig_atom_indices_in_rsub = list(map(lambda x: GetAtomWithAtomMapNum(rsig, x).GetIdx(), rsub_atom_indices))

        # Corresponding atoms in mol should have neighbors inside rsig_matches
        def verify(match):
            neighbors = []
            for idx in rsig_atom_indices_in_rsub:
                corr_idx = match[idx]
                atom = mol.GetAtomWithIdx(corr_idx)
                neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
            if not set(neighbors) - set(match):
                return True
            return False

        rsig_matches = list(filter(verify, rsig_matches))
        return rsig_matches

    def apply_action(self, input_mol, rsub, rcen, rsig, rsig_cs_indices, psub, pcen, psig, psig_cs_indices):
        # Some basic conversions acc. to dataset format
        input_mol = Chem.Mol(input_mol)
        rsig = Chem.MolFromSmiles(rsig)
        psig = Chem.MolFromSmiles(psig)
        rsig_cs_indices = list(map(int, rsig_cs_indices.split(".")))
        psig_cs_indices = list(map(int, psig_cs_indices.split(".")))

        # Find rsig in input_mol
        rsig_matches = input_mol.GetSubstructMatches(rsig)

        # If multiple matches, choose one where rsub/rcen makes sense
        if len(rsig_matches) > 1:
            rsig_matches = self._filter_sensible_rsig_matches(input_mol, rsig_matches, rsig, Chem.MolFromSmiles(rsub), rcen)

        # FIXME: Provide option to use more than just the first match
        rsig_match = rsig_matches[0]

        # Find indices to be exchanged
        input_mol_cs_indices = np.array(rsig_match)[rsig_cs_indices].tolist()

        # Exchange indices (replace bonds for atoms at given indices)
        rwmol = Chem.RWMol(mol_with_atom_index(input_mol))
        num_atoms = input_mol.GetNumAtoms()
        new_psig = Chem.Mol(psig)
        for atom in new_psig.GetAtoms():
            atom.SetAtomMapNum(atom.GetAtomMapNum() + num_atoms)
        rwmol.InsertMol(Chem.Mol(new_psig))

        rsig_cs_atom_map_num = list(input_mol_cs_indices)
        psig_cs_atom_map_num = (np.array(psig_cs_indices) + num_atoms).tolist()

        for r_an, p_an in zip(rsig_cs_atom_map_num, psig_cs_atom_map_num):
            r_idx = GetAtomWithAtomMapNum(rwmol, r_an).GetIdx()
            p_idx = GetAtomWithAtomMapNum(rwmol, p_an).GetIdx()
            for conn in find_connecting_atoms_not_in_sig(input_mol, rsig_match, r_idx):
                rwmol.AddBond(p_idx, conn, input_mol.GetBondBetweenAtoms(r_idx, conn).GetBondType())
                rwmol.RemoveBond(r_idx, conn)

        # Remove the atoms from rsig
        for atm_num in rsig_match:
            rwmol.RemoveAtom(GetAtomWithAtomMapNum(rwmol, atm_num).GetIdx())

        mol = mol_without_atom_index(Chem.Mol(rwmol))
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
            mol = Chem.MolFromSmiles(clean_hydrogen_in_smiles(Chem.MolToSmiles(mol)))
        else:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

        assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None, "Final mol is not valid"
        assert "." not in Chem.MolToSmiles(mol), "More than 1 molecule in result"
        return mol


# Module-level singleton + convenience functions for backwards compatibility and multiprocessing
_default_action_space = None


def get_default_action_space():
    global _default_action_space
    if _default_action_space is None:
        _default_action_space = ActionSpace()
    return _default_action_space


def get_applicable_actions(mol, random_state=None):
    return get_default_action_space().get_applicable_actions(mol, random_state)


def get_random_action(mol, random_state=None):
    return get_default_action_space().get_random_action(mol, random_state)


def apply_action(input_mol, rsub, rcen, rsig, rsig_cs_indices, psub, pcen, psig, psig_cs_indices):
    return get_default_action_space().apply_action(input_mol, rsub, rcen, rsig, rsig_cs_indices, psub, pcen, psig, psig_cs_indices)
