# Modified from https://github.com/ziqi92/Modof
"""
iclr19-graph2graph

Copyright (c) 2019 Wengong Jin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Modified from https://github.com/wengong-jin/iclr19-graph2graph
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
from reactionrl.rewards import drd2_scorer
from reactionrl.rewards import sascorer

def similarity(amol, bmol, sim_type=None):
    if amol is None or bmol is None:
        return 0.0

    if sim_type == "binary":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    else:
        fp1 = AllChem.GetMorganFingerprint(amol, 2, useChirality=False)
        fp2 = AllChem.GetMorganFingerprint(bmol, 2, useChirality=False)

    sim = DataStructs.TanimotoSimilarity(fp1, fp2)

    return sim

def drd2(mol):
    if mol is None:
        return 0.0
    return drd2_scorer.get_score(mol)

def qed(mol):
    if mol is None:
        return 0.0
    return QED.qed(mol)

def logP(mol):
    if mol is None:
        return 0.0
    return Descriptors.MolLogP(mol)

def SA(mol):
    if mol is None:
        return 0.0
    return sascorer.calculateScore(mol)
