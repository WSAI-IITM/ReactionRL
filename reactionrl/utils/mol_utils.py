"""Shared molecular utility functions.

RDKit helper functions used across the package for molecule manipulation,
fingerprinting, and atom mapping.
"""
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import numpy as np


def get_mol_certificate(mol):
    '''
    Takes a Chem.Mol and returns Morgan fingerprint in base64
    '''
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBase64()


def clean_hydrogen_in_smiles(smiles):
    '''
    Some clean-ups Idk how to do in molecule. So I do it in smiles after conversion.
    1. Remove extra hydrogens for even sized rings (odd sized rings require one atom with explicitly competed valency: like c1cc[nH]c1)
    2. Sometimes in the odd sized rings, there is the valency is extra for the explicitly completed atom - try removing hydrogen for those
    '''
    smiles = re.sub(r"\[([a-zA-Z])H[0-9]\]", r"\1", smiles)

    if Chem.MolFromSmiles(smiles) is None:
        smiles = re.sub(r"\[([a-zA-Z])H\]", r"\1", smiles)

    return smiles


def mol_with_atom_index(mol):
    '''
    draw molecule with index
    '''
    colored = False
    if hasattr(mol, "__sssAtoms"):
        sss = mol.__sssAtoms
        colored = True
    mol = Chem.Mol(mol)
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    if colored:
        mol.__sssAtoms = sss
    return mol


def smiles_without_atom_index(smiles):
    '''
    Convert smiles with numbers to smiles without numbers
    '''
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol)


def mol_without_atom_index(mol):
    '''
    Convert smiles with numbers to smiles without numbers
    '''
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).ClearProp('molAtomMapNumber')
    return mol


def find_connecting_atoms_not_in_sig(mol, sig_indices, centre):
    cen_atom = mol.GetAtomWithIdx(centre)
    neighbors_indices = list(map(lambda x: x.GetIdx(), cen_atom.GetNeighbors()))
    return set(neighbors_indices) - set(sig_indices)


def GetAtomWithAtomMapNum(mol, num):
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == num:
            return atom
    return None
