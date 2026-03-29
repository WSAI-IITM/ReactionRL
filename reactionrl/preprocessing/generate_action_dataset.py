from rdkit import Chem
import numpy as np
import pandas as pd
import tqdm
import re

try:
    from IPython.display import display
except ImportError:
    display = print  # fallback for non-IPython environments


def add_immediate_neighbors(mol, indices, add_aromatic_cycles=True):
    """Add immediate neighbor atoms to a set of indices.

    If add_aromatic_cycles is true, adds the whole aromatic cycle for any
    newly added aromatic atom. Returns (indices, added_aromatic_cycle) when
    add_aromatic_cycles is True, otherwise just indices.
    """
    def _add_neighbors(idx_list):
        atoms = list(map(lambda x: mol.GetAtomWithIdx(int(x)), idx_list))
        neighbors = []
        for atom in atoms:
            neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
        return np.unique(neighbors).tolist()

    new_indices = _add_neighbors(indices)

    if set(new_indices) == set(indices):
        return indices

    added_aromatic_cycle = False
    if add_aromatic_cycles:
        if any(list(map(lambda idx: mol.GetAtomWithIdx(idx).GetIsAromatic(), list(set(new_indices) - set(indices))))):
            indices = list(new_indices)
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

dataset = pd.read_csv("datasets/my_uspto/simulator_dataset.csv", index_col=0)

def remove_h(smiles): 
    return re.sub("H[0-9]?", "", smiles)

def cure_ending_aromatic_atom(smiles):
    matches = re.findall(r"\[[^\]]+\]", smiles)
    idx_to_fix = 0
    if len(matches) == 1:
        if matches[0][1].islower():
            pass # Fix idx 0
        else:
            return smiles # Do nothing
    else:
        if matches[0][1].islower():
            if matches[-1][1].islower():
                return smiles # both lower - do nothing
            else:
                pass # first lower - fix idx 0
        else:
            if matches[-1][1].islower():
                idx_to_fix = -1 # last lower - fix idx -1
            else:
                return smiles # both upper - do nothing
    
    matches[idx_to_fix] = matches[idx_to_fix][:1] + matches[idx_to_fix][1].upper() + matches[idx_to_fix][2:]
    return "".join(matches)

# First we fix 2 things -
# 1. If the signature has part of aromatic ring, it doesn't create a valid molecule. So we "capitalize" the ending atoms if they are small (aromatic)
# 2. Sometimes there are extra or less H's. Let the Chem.MolFromSmiles automatically determine the H's
rsub_worked = np.array(list(map(lambda x: pd.isna(x) or Chem.MolFromSmiles(x) is not None, dataset["rsub"].tolist())))
psub_worked = np.array(list(map(lambda x: pd.isna(x) or Chem.MolFromSmiles(x) is not None, dataset["psub"].tolist())))
rsig_worked = np.array(list(map(lambda x: pd.isna(x) or Chem.MolFromSmiles(x) is not None, dataset["rsig"].tolist())))
psig_worked = np.array(list(map(lambda x: pd.isna(x) or Chem.MolFromSmiles(x) is not None, dataset["psig"].tolist())))

dataset.loc[~rsub_worked, "rsub"] = list(map(cure_ending_aromatic_atom, dataset[~rsub_worked]["rsub"]))
dataset.loc[~psub_worked, "psub"] = list(map(cure_ending_aromatic_atom, dataset[~psub_worked]["psub"]))
dataset.loc[~rsig_worked, "rsig"] = list(map(remove_h, dataset[~rsig_worked]["rsig"]))
dataset.loc[~psig_worked, "psig"] = list(map(remove_h, dataset[~psig_worked]["psig"]))

# First we remove those reactions where the signature extraction didnt work properly 
# In these cases, the Chem.Mol of either of rsig or psig is None
rsub_worked = np.array(list(map(lambda x: pd.isna(x) or Chem.MolFromSmiles(x) is not None, dataset["rsub"].tolist())))
psub_worked = np.array(list(map(lambda x: pd.isna(x) or Chem.MolFromSmiles(x) is not None, dataset["psub"].tolist())))

dataset = dataset[rsub_worked & psub_worked]

rsig_worked = np.array(list(map(lambda x: pd.isna(x) or Chem.MolFromSmiles(x) is not None, dataset["rsig"].tolist())))
psig_worked = np.array(list(map(lambda x: pd.isna(x) or Chem.MolFromSmiles(x) is not None, dataset["psig"].tolist())))

dataset = dataset[rsig_worked & psig_worked]
print(dataset.shape)

# we only consider actions from the following reactions: 
# 1. len(rcen) >= len(pcen) (Hydrogen in product is allowed)
# 2. len(rcen) != 0. len(pcen) == 0 is allowed. = 0 means Hydrogen, so we do not want Hydrogen removals because
#    there are too many options in that case. Adding a Hydrogen is fine since that is deterministic.
# 3. len(rcen) <= 2 and len(pcen) <= 2. The cases where this is not true is typically because the 
#    reactant-product mapping is wrong. These cases are very few anyways so it's fine.
rlen = np.array(list(map(lambda x: len(x.strip("[]").split(",")) if len(x) > 2 else 0, dataset["rcen"])))
plen = np.array(list(map(lambda x: len(x.strip("[]").split(",")) if len(x) > 2 else 0, dataset["pcen"])))

dataset = dataset[(rlen>=plen) & (rlen!=0) & (rlen<=2) & (plen<=2)]
print(dataset.shape)

# Mare sure no hydrogens in reactions. For product, add hydrogen to the actions
assert not dataset["rsig"].isna().any(), "There are hydrogens in reactant signatures!!!!"

# In the previous script, I forgot to add atom number to hydrogen. Let's do that too.
dataset.loc[dataset["rsub"]=="[H]"]["rsig"] = "[H:0]"
dataset.loc[dataset["rsub"]=="[H]"]["rsub"] = "[H:0]"

dataset.loc[dataset["psig"].isna(), "pcen"] = "[0]"
dataset.loc[dataset["psig"].isna(), "psub"] = "[H:0]"
dataset.loc[dataset["psig"].isna(), "psig"] = "[H:0]"

# Taking a look at what combinations are left
rlen = np.array(list(map(lambda x: len(x.strip("[]").split(",")) if len(x) > 2 else 0, dataset["rcen"])))
plen = np.array(list(map(lambda x: len(x.strip("[]").split(",")) if len(x) > 2 else 0, dataset["pcen"])))
np.unique(list(map(lambda x, y: f"{x}-{y}", rlen, plen)), return_counts=True)

# temp_df = dataset[(rlen!=1) & (plen!=1)]
temp_df = dataset[(rlen!=2) & (plen!=2)]

# First let us remove those cases where number of centres is more than number of signatures
rcenlen = np.array(list(map(lambda x: len(x.split(",")), temp_df["rcen"])))
pcenlen = np.array(list(map(lambda x: len(x.split(",")), temp_df["pcen"])))
rsiglen = np.array(list(map(lambda x: len(x.split(".")), temp_df["rsig"])))
psiglen = np.array(list(map(lambda x: len(x.split(".")), temp_df["psig"])))

temp_df = temp_df[(rcenlen==rsiglen) & (pcenlen==psiglen)]

# Empty bonds (due to Hydrogen) - fix 'em
dataset.loc[(dataset["rbond"] == "[]") | (dataset["rbond"] == "[[]]"), "rbond"] = "[[1.0]]"
dataset.loc[(dataset["pbond"] == "[]") | (dataset["pbond"] == "[[]]"), "pbond"] = "[[1.0]]"

rbond_len = np.array(list(map(len, dataset["rbond"])))
pbond_len = np.array(list(map(len, dataset["pbond"])))

temp_df = dataset[rbond_len != pbond_len]

# Get 1-1 actions
dataset = dataset[(rlen==1)&(plen==1)]

print(dataset.shape)

# First, we confirm that num(sig) = num(cen) = num(sub)
rcenlen = np.array(list(map(lambda x: len(x.split(",")), dataset["rcen"])))
pcenlen = np.array(list(map(lambda x: len(x.split(",")), dataset["pcen"])))
rsiglen = np.array(list(map(lambda x: len(x.split(".")), dataset["rsig"])))
psiglen = np.array(list(map(lambda x: len(x.split(".")), dataset["psig"])))
rsublen = np.array(list(map(lambda x: len(x.split(".")), dataset["rsub"])))
psublen = np.array(list(map(lambda x: len(x.split(".")), dataset["psub"])))

dataset = dataset[(rcenlen==rsiglen) & (pcenlen==psiglen) & (rcenlen==rsublen) & (pcenlen==psublen)]
print(dataset.shape)

# ACTION REMAP  
# Renumber the atoms in the actions. Renumber the centre.  
# Find which rsig are same(and the same corresponding centre), then map unique rsig to all possible psigs 
# Renumber the atoms in the signatures to start from 0 and be contiguous. Renumber the centre accordingly. 

def reduce_atom_num_and_centre(sub, sig, cen):
    '''
    Reduce the atom numbers to start from 0. Reduce centre by same amount. 
    Centre is in format '[x]'. Make it x of type int.
    Return new_sub, new_sig, x
    '''
    mol = Chem.MolFromSmiles(sig)
    sub_mol = Chem.MolFromSmiles(sub)
    
    atom_num_list = list(map(lambda x: int(mol.GetAtomWithIdx(x).GetProp("molAtomMapNumber")), range(mol.GetNumAtoms())))
    atom_num_list = np.array(atom_num_list)
    
    sub_num_list = list(map(lambda x: int(sub_mol.GetAtomWithIdx(x).GetProp("molAtomMapNumber")), range(sub_mol.GetNumAtoms())))
    sub_num_list = np.array(sub_num_list)
    
    for idx in range(mol.GetNumAtoms()):
        # First replace that num in sub
        atm_num = int(mol.GetAtomWithIdx(idx).GetProp("molAtomMapNumber"))
        if atm_num in sub_num_list:
            sub_mol.GetAtomWithIdx(int(abs(sub_num_list-atm_num).argmin())).SetProp("molAtomMapNumber", str(idx))

        # If centre matches, update
        if atm_num == int(cen[1:-1]):
            new_cen = abs(atom_num_list-int(cen[1:-1])).argmin()
        
        # Then change in sig
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str(idx) )
    
    return Chem.MolToSmiles(sub_mol), Chem.MolToSmiles(mol), new_cen

i = 4
reduce_atom_num_and_centre(dataset.iloc[i]["psub"], dataset.iloc[i]["psig"], dataset.iloc[i]["pcen"])

for i in tqdm.tqdm(range(dataset.shape[0]), total=dataset.shape[0]):
    _, _, rsig, psig, rsub, psub, rcen, pcen, _, _ = dataset.iloc[i]
    rsub, rsig, rcen = reduce_atom_num_and_centre(rsub, rsig, rcen)
    psub, psig, pcen = reduce_atom_num_and_centre(psub, psig, pcen)
    dataset.iloc[i]["rsub"] = rsub
    dataset.iloc[i]["rsig"] = rsig
    dataset.iloc[i]["psub"] = psub
    dataset.iloc[i]["psig"] = psig
    dataset.iloc[i]["rcen"] = rcen
    dataset.iloc[i]["pcen"] = pcen

# Certifications
from rdkit.Chem import AllChem

def get_mol_certificate(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBase64()

# First create clusters based on atom type and numbers
d = {}
for i in tqdm.tqdm(range(dataset.shape[0]), total=dataset.shape[0]):
    rsig = dataset.iloc[i]["rsig"]
    
    l = list(map(str.upper, list(filter(str.isalpha, list(rsig)))))
    l.sort()
    _id = "".join(l)
    _id = _id.replace("H", "")
    
    if _id in d:
        d[_id].append(i)
    else:
        d[_id] = [i]

# Divide the clusters based on signature structure
def divide_based_on_signatures(listy, r_or_p="r"):
    if r_or_p=="r":
        sig_choice = "rsig"
    elif r_or_p == "p":
        sig_choice = "psig"
        
    reference_d = {}
    cluster_d = {}
    certi_d = {}
    for i in listy:
        rsig = Chem.MolFromSmiles(dataset.iloc[i][sig_choice])
        if not reference_d: # Starting element
            cluster_d[i] = [i]
            reference_d[i] = rsig
            continue
        
        # Search for appropriate cluster
        clustered = False
        for key_i in reference_d:
            ref_rsig = reference_d[key_i]
            ref_smile = Chem.MolToSmiles(ref_rsig)
            if ref_smile not in certi_d:
                certi_d[ref_smile] = get_mol_certificate(ref_rsig)
            if rsig.HasSubstructMatch(ref_rsig) and ref_rsig.HasSubstructMatch(rsig) and get_mol_certificate(rsig) == certi_d[ref_smile]: # Because using just 1 can give wrong results - Imagine my frustration at finding this out, RDKit developers...
                # Add to this cluster
                cluster_d[key_i].append(i)
                clustered = True
                break
                
        if not clustered: # Add a new cluster
            cluster_d[i] = [i]
            reference_d[i] = rsig
            
    return cluster_d

rsig_d = {}
for key in tqdm.tqdm(d, total=len(d.keys())):
    rsig_d.update(divide_based_on_signatures(d[key]))

# Divide the clusters based on reaction centre
def divide_based_on_centre(listy, r_or_p="r"):
    if r_or_p=="r":
        sig_choice = "rsig"
        cen_choice = "rcen"
    elif r_or_p == "p":
        sig_choice = "psig"
        cen_choice = "pcen"
        
    reference_d = {}
    reference_cen = {}
    cluster_d = {}
    for i in listy:
        rsig = Chem.MolFromSmiles(dataset.iloc[i][sig_choice])
        if not reference_d: # Starting element
            cluster_d[i] = [i]
            reference_d[i] = rsig
            reference_cen[i] = dataset.iloc[i][cen_choice]
            continue
        
        # Search for appropriate cluster
        clustered = False
        for key_i in reference_d:
            ref_rsig = reference_d[key_i]
            if rsig.GetSubstructMatch(ref_rsig)[reference_cen[key_i]] == dataset.iloc[i][cen_choice]:
                # Add to this cluster
                cluster_d[key_i].append(i)
                clustered = True
                break
                
        if not clustered: # Add a new cluster
            cluster_d[i] = [i]
            reference_d[i] = rsig
            reference_cen[i] = dataset.iloc[i][cen_choice]
            
    return cluster_d

rcen_d = {}
for key in tqdm.tqdm(rsig_d, total=len(rsig_d.keys())):
    rcen_d.update(divide_based_on_centre(rsig_d[key]))

# Next we divide based on psig, pcen and pbond so that we can get unique rsig to unique psig matchings
psig_d = {}
for key in tqdm.tqdm(rcen_d, total=len(rcen_d.keys())):
    psig_d[key] = divide_based_on_signatures(rcen_d[key], r_or_p="p")    

pcen_d = {}
for key in tqdm.tqdm(psig_d, total=len(psig_d.keys())):
    new_d = {}
    temp_d = psig_d[key]
    for temp_key in temp_d:
        new_d[temp_key] = divide_based_on_centre(temp_d[temp_key], r_or_p="p")
    pcen_d[key] = new_d

temp_d = dict(pcen_d)
pcen_d = {}
for key in temp_d:
    pcen_d[key] = {}
    d = temp_d[key]
    for key2 in d:
        pcen_d[key].update(d[key2])
        
del temp_d

# Finally, we dump a DataFrame with unique rsig-psig pairs(actions)
# Also, we group rsigs so that searching becomes easier later on
unique_rsig_indexer = np.zeros(shape=dataset.shape[0], dtype=int)
for key in rcen_d:
    for ele in rcen_d[key]:
        unique_rsig_indexer[ele] = key
        
dataset["rsig_clusters"] = unique_rsig_indexer

unique_action_indexer = []
for key in pcen_d:
#     for key2 in pcen_d[key]:
    unique_action_indexer.extend(list(pcen_d[key].keys()))

dataset = dataset.iloc[unique_action_indexer]

# I had done this before but some are still there somehow
dataset = dataset[dataset["rbond"]!="[[]]"]
dataset = dataset[dataset["pbond"]!="[[]]"]

from rdkit.Chem import rdFMCS

def mol_with_atom_index( mol ):
    '''
    draw molecule with index
    '''
    colored = False
    if hasattr(mol, "__sssAtoms"):
        sss = mol.__sssAtoms
        colored = True
    mol = Chem.Mol(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    if colored:
        mol.__sssAtoms = sss
    return mol

def smiles_without_atom_index( smiles ):
    '''
    Convert smiles with numbers to smiles without numbers
    '''
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).ClearProp( 'molAtomMapNumber' )
    return Chem.MolToSmiles(mol)

def mol_without_atom_index(mol):
    '''
    Convert smiles with numbers to smiles without numbers
    '''
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).ClearProp( 'molAtomMapNumber' )
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

def get_cs_indices(input_mol, psig, rsig_match, rcen, pcen, debug=False):
    '''
    Find mcs between mol1 and mol2 and get indices in mol1 and mol2 corresponding to the cs 
    Returns (mol1 indices), (mol2 indices)
    '''
    # find mcs
    mcs = rdFMCS.FindMCS([input_mol, psig])
    cs = Chem.MolFromSmarts(mcs.smartsString)
    
    if debug:
        print("Input mol and psig")
        display(input_mol)
        display(psig)
        print("cs")
        display(cs)
        
    # Get initial cs indices
    input_mol_cs_indices = np.array(rsig_match)[list(Chem.Mol(rsig).GetSubstructMatch(cs))].tolist()
    if not input_mol_cs_indices:
        input_mol_cs_indices = np.array(rsig_match)[list(Chem.Mol(rsig).GetSubstructMatch(cs))].tolist()
    psig_cs_indices = Chem.Mol(psig).GetSubstructMatches(cs)
    
    if debug:
        print("psig_cs_indices(before)", psig_cs_indices)
    
    # if there's more than 1 match, choose the one which has one 1 neighbor
    if len(psig_cs_indices) == 1:
        psig_cs_indices = psig_cs_indices[0]
    else:
        if debug:
            print(list(map(lambda x: (set(add_immediate_neighbors(psig, x, add_aromatic_cycles=False)), set(x)), psig_cs_indices)))
        psig_cs_indices = list(filter(lambda x: len(set(add_immediate_neighbors(psig, x, add_aromatic_cycles=False)) - set(x))==1, psig_cs_indices))[0]
    
    if debug:
        print("psig_cs_indices(after)", psig_cs_indices)
    
    if debug:
        display(mol_with_atom_index(input_mol))
    
    # If CS is only a benzene ring, the substructure matching can give a wrong ordering due to the symmetric nature of benzene. 
    # So I find the atom connecting to the centre and rotate one of the indices to match the other
    if Chem.MolToSmarts(cs) == '[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1':
        if debug:
            print("CS is benzene. Doing extra calculations for index matching...")
        def connecting_atom_idx_in_indices(mol, cen, indices):
            cen_atom = mol.GetAtomWithIdx(int(cen))
            cen_neighbors = list(map(lambda x: x.GetIdx(), cen_atom.GetNeighbors()))
            connecting_atoms = set(cen_neighbors).intersection(set(indices))
            assert len(connecting_atoms) == 1, f"Expecting 1 connecting atom. Found {len(connecting_atoms)}"
            return list(connecting_atoms)[0]
        
        in_mol_connecting_idx = connecting_atom_idx_in_indices(input_mol, rsig_match[rcen], input_mol_cs_indices)
        psig_connecting_idx = connecting_atom_idx_in_indices(psig, pcen, psig_cs_indices)
        
        a = np.argmin(np.abs(np.array(input_mol_cs_indices) - in_mol_connecting_idx))
        b = np.argmin(np.abs(np.array(psig_cs_indices) - psig_connecting_idx))
        
        if debug:
            print(in_mol_connecting_idx, "<--->", psig_connecting_idx)
            print("Rotating....")
            print(f"a={a}, b={b}")
            print("Before")
            print(input_mol_cs_indices)
            print(psig_cs_indices)

        if a < b:
            # left rotate psig idx
            psig_cs_indices = psig_cs_indices[b-a:] + psig_cs_indices[:b-a]
        elif b < a:
            # left rotate in_mol idx
            input_mol_cs_indices = input_mol_cs_indices[a-b:] + input_mol_cs_indices[:a-b]
        
        if debug:
            print("After")
            print(input_mol_cs_indices)
            print(psig_cs_indices)
    
    return input_mol_cs_indices, psig_cs_indices
    
# For psig="[H:0]", MCS doesn't work. So for now, I'm removing all such actions
# The easiest way to deal with these actions is to have another routine for "apply_action" which removes rsub (there's nothing to add since psub is Hydrogen)
dataset = dataset[dataset["psig"]!="[H:0]"]
print(dataset.shape)

dataset.loc[:, "rsig_cs_indices"] = [-1] * dataset.shape[0]
dataset.loc[:, "psig_cs_indices"] = [-1] * dataset.shape[0]

# Store cs indices in dataset so that we don't have to compute every time
for i in tqdm.tqdm(range(0, dataset.shape[0])):
    rsig = Chem.MolFromSmiles(dataset.iloc[i]["rsig"])
    psig = Chem.MolFromSmiles(dataset.iloc[i]["psig"])
    rsig_match = list(range(rsig.GetNumAtoms()))
    rcen = dataset.iloc[i]["rcen"]
    pcen = dataset.iloc[i]["pcen"]
    
    try:
        r_cs, p_cs = get_cs_indices(rsig, psig, rsig_match, rcen, pcen)
        dataset["rsig_cs_indices"].iloc[i] = ".".join(list(map(str, r_cs)))
        dataset["psig_cs_indices"].iloc[i] = ".".join(list(map(str, p_cs)))
    except Exception as e:
        pass

dataset = dataset[dataset["rsig_cs_indices"] != -1]
print("New shape for dataset:", dataset.shape)

print("Dumping...")
dataset.to_csv("datasets/my_uspto/action_dataset.csv")