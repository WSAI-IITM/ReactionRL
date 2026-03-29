import numpy as np
from rdkit import Chem

from deepchem.utils.typing import RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.feat.molecule_featurizers.mol_graph_conv_featurizer import _construct_atom_feature, _construct_bond_feature


def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
    """Calculate molecule graph features from RDKit mol object.

    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit mol object.

    Returns
    -------
    graph: GraphData
      A molecule graph with some features.
    """
    if 'mol' in kwargs:
        datapoint = kwargs.get("mol")
        raise DeprecationWarning(
            'Mol is being phased out as a parameter, please pass "datapoint" instead.'
        )

    if self.use_partial_charge:
        try:
            datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
        except:
        # If partial charges were not computed
            try:
                from rdkit.Chem import AllChem
                AllChem.ComputeGasteigerCharges(datapoint)
            except ModuleNotFoundError:
                raise ImportError("This class requires RDKit to be installed.")

    # construct atom (node) feature
    h_bond_infos = construct_hydrogen_bonding_info(datapoint)
    atom_features = np.asarray(
        [
            _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                    self.use_partial_charge)
            for atom in datapoint.GetAtoms()
        ],
        dtype=float,
    )

    # If single atom molecule, add a dummy atom with '0' features
    if atom_features.shape[0] == 1:
        atom_features = np.concatenate([atom_features, np.zeros(atom_features.shape)])

        # Now bond information can be created from any X-X molecule
        datapoint = Chem.MolFromSmiles("CC")

    # construct edge (bond) index
    src, dest = [], []
    for bond in datapoint.GetBonds():
        # add edge list considering a directed graph
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [start, end]
        dest += [end, start]

    # construct edge (bond) feature
    bond_features = None  # deafult None
    if self.use_edges:
        features = []
        for bond in datapoint.GetBonds():
            features += 2 * [_construct_bond_feature(bond)]
        bond_features = np.asarray(features, dtype=float)

    return GraphData(
        node_features=atom_features,
        edge_index=np.asarray([src, dest], dtype=int),
        edge_features=bond_features)
