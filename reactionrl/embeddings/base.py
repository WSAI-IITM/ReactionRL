"""Base class for molecular embedding models.

Subclasses must implement mol_to_embedding() and atom_to_embedding()
to provide graph-level and atom-level molecular representations.
"""
from abc import ABC, abstractmethod


class BaseEmbeddingClass(ABC):
    """Abstract base class for molecule embedders.

    Implementations include:
    - Zinc_GIN_Embedder: GIN model pretrained on ZINC 2M
    - ChemBL_MPNN_Embedder: MPNN model pretrained on ChemBL
    """

    @abstractmethod
    def mol_to_embedding(self, mol):
        """Compute a graph-level embedding for a molecule.

        Args:
            mol: RDKit Mol object or SMILES string.

        Returns:
            Embedding tensor/array.
        """
        raise NotImplementedError()

    @abstractmethod
    def atom_to_embedding(self, mol, idx):
        """Compute an atom-level embedding.

        Args:
            mol: RDKit Mol object or SMILES string.
            idx: Atom index within the molecule.

        Returns:
            Embedding tensor/array for the specified atom.
        """
        raise NotImplementedError()
