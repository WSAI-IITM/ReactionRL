from rdkit import Chem
import numpy as np
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print  # fallback for non-IPython environments
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import rdFMCS

# draw molecule with index
def mol_with_atom_index( mol ):
    mol = deepcopy(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def highlight_atoms(mol, hit_ats):
    '''
    Highlight the atoms in mol that have index in 'hit_ats'
    '''
#     # this is the code given in rdkit docs but doesn't actually work
#     d = rdMolDraw2D.MolDraw2DSVG(500, 500) # or MolDraw2DCairo to get PNGs
#     rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,)
    mol.__sssAtoms = hit_ats # workaround for now. Might not work in a later version of rdkit
class RLMol:
    def __init__(self, mol):
        self.mol = mol
    
    def display_mol(self, atom_num=False, highlight=None):
        mol = Chem.Mol(self.mol)
        if atom_num:
            mol = mol_with_atom_index(mol)
        if highlight=="sub":
            highlight_atoms(mol, self.sub)
        if highlight=="sig":
            highlight_atoms(mol, self.sig)
        display(mol)
        
    def calc_sig(self):
        self.sig = list(self.sub)
            
        for _ in range(2):
            temp_list = []
            for n in self.sig:
                temp_list.extend([atom.GetIdx() for atom in self.mol.GetAtomWithIdx(n).GetNeighbors()])
            self.sig.extend(temp_list)
            self.sig = list(set(self.sig))
        
        # Add any aromatic rings if attached
        self.sig = self.add_aromatic_ring_if_possible(self.sig)
        
        self.sig = np.unique(self.sig).tolist()
    
    def calculate_centres_and_signatures(self, common_subsequence, debug=False):
        # input
        mol = Chem.Mol(self.mol)
        self.common_subsequence = common_subsequence
        cs = Chem.Mol(self.common_subsequence)
        
        # deal with atom indices
        mol_indices = list(range(mol.GetNumAtoms()))
        mol_indices_in_cs = Chem.Mol(mol).GetSubstructMatch(cs)
        if not mol_indices_in_cs: # in some rare cases, GetSubstructureMatch returns empty tuple - rdkit algo problem
            cs = rdFMCS.FindMCS([mol, cs])
            self.common_subsequence = Chem.MolFromSmarts(cs.smartsString)
            cs = Chem.Mol(self.common_subsequence)
#             display(mcs)
            mol_indices_in_cs = Chem.Mol(mol).GetSubstructMatch(cs)
            
        
        # find signature
        difference = list(set(mol_indices) - set(mol_indices_in_cs))
        self.sub = difference
        
        # find centre (and bond)
        self.cen = []
        for idx in self.sub:
            atom = mol.GetAtomWithIdx(idx)
            neighbors = atom.GetNeighbors()
            neighbors_indices = list(map(lambda x: x.GetIdx(), neighbors))
            if set(neighbors_indices) - set(self.sub): # this atom has a neighbor outside of signature
                self.cen.append(idx)
        
        # find bond
        self.calc_bond()
        
        # find sig
        self.calc_sig()

        # if debug, display
        if debug:
            print("Subgraph")
            self.display_mol(atom_num=True, highlight="sub")
            print("Signature")
            self.display_mol(atom_num=True, highlight="sig")
            
            print("Centre at", self.cen)
    
    def try_to_merge(self, debug=False):
        ''' 
        if more than 1 centre, tries to merge them into 1 if they have a common atom
        Returns True if merge happens, else False
        '''
        mol = Chem.Mol(self.mol)
        cs = Chem.Mol(self.common_subsequence)
        if len(self.cen) < 2:
            return False
        
        conn_atom_d = {}
        conn_idx_list = []
        mol_indices_in_cs = Chem.Mol(mol).GetSubstructMatch(cs)
        for cen in self.cen:
            cen_atom = mol.GetAtomWithIdx(cen)
            cen_neighbors_indices = list(map(lambda x: x.GetIdx(), cen_atom.GetNeighbors()))
            
            connecting_atom_idx = list(set(cen_neighbors_indices).intersection(set(mol_indices_in_cs)))[0]
            connecting_atom = mol.GetAtomWithIdx(connecting_atom_idx)
            
            conn_atom_d[cen] = [connecting_atom]
            conn_idx_list.append(connecting_atom_idx)
            
        # if they have common atom, merge
        if len(np.unique(conn_idx_list)) == 1:
            # If connecting atom is aromatic, don't merge
            if mol.GetAtomWithIdx(conn_idx_list[0]).GetIsAromatic():
                return False
            self.sub.append(conn_idx_list[0])
            self.cen = [conn_idx_list[0]]
            self.calc_bond()
            self.calc_sig()
            
            # if debug, display
            if debug:
                print("Subgraph")
                self.display_mol(atom_num=True, highlight="sub")
                print("Signature")
                self.display_mol(atom_num=True, highlight="sig")

                print("Centre at", self.cen)
            
            return True
        return False
    
    def get_centre_in_cs(self, merge=False):
        '''
        Get the index in cs that is equivalent to centre in mol
        
        If this is for a merge operation, then re
        '''
        mol = Chem.Mol(self.mol)
        cs = Chem.Mol(self.common_subsequence)
        
        mol_indices_in_cs = Chem.Mol(mol).GetSubstructMatch(cs)
        
        if merge:
            connecting_atom_idx = self.cen[0]
        else:
            try:
                # Get the atom index in cs connected to centre
                cen_neighbors_indices = list(map(lambda x: x.GetIdx(), mol.GetAtomWithIdx(self.cen[0]).GetNeighbors()))
                connecting_atom_idx = list(set(cen_neighbors_indices).intersection(set(mol_indices_in_cs)))[0]
            except Exception as e:
                connecting_atom_idx = self.cen[0]
        
        mol_indices_in_cs = np.array(mol_indices_in_cs)
        return int(abs(mol_indices_in_cs - connecting_atom_idx).argmin())
        
    def calc_bond(self):
        self.bond = []
        for c in self.cen:
            atom = self.mol.GetAtomWithIdx(c)
            neighbors = atom.GetNeighbors()
            neighbors_indices = list(map(lambda x: x.GetIdx(), neighbors))
            self.bond.append([self.mol.GetBondBetweenAtoms(c, natom).GetBondTypeAsDouble() for natom in set(neighbors_indices) - set(self.sub)])
    
    def push_back_centre(self, cs_centre, debug):
        '''
        This function adds the adjoining atom from common subsequence (cs) to the signature
        In case singature is null, cs_centre is used to determine the atom to be added.
        '''
        mol = Chem.Mol(self.mol)
        cs = Chem.Mol(self.common_subsequence)
        
        if self.cen:
            cen = self.cen[0]
            cen_atom = mol.GetAtomWithIdx(cen)
            cen_neighbors_indices = list(map(lambda x: x.GetIdx(), cen_atom.GetNeighbors()))
            mol_indices_in_cs = Chem.Mol(mol).GetSubstructMatch(cs)
            connecting_atom_idx = list(set(cen_neighbors_indices).intersection(set(mol_indices_in_cs)))[0]

            self.sub.append(connecting_atom_idx)
            self.cen = [connecting_atom_idx]

        else:
            # calculate the reaction centre from common subsequence and use it for signature
            self.cen = [int(Chem.Mol(mol).GetSubstructMatch(cs)[cs_centre])]
            self.sub = list(self.cen)
        
        self.calc_bond()
        self.calc_sig()
        
        # if debug, display
        if debug:
            print("Subgraph")
            self.display_mol(atom_num=True, highlight="sub")
            print("Signature")
            self.display_mol(atom_num=True, highlight="sig")
            
            print("Centre at", self.cen)
            
    def get_subgraph(self):
        # calc Mol from list of ints
        sub = None
        mol = mol_with_atom_index(self.mol)
        with Chem.RWMol(mol) as mw:
            for idx in set(list(range(self.mol.GetNumAtoms()))) - set(self.sub):
                mw.RemoveAtom(idx)
            sub = Chem.Mol(mw)
        return mw
    
    def get_smiles_subgraph(self):
        return Chem.MolToSmiles(self.get_subgraph())
    
    
    def add_aromatic_ring_if_possible(self, atom_indices): 
        # Aromatic atoms cannot be represented individually in signatures so we add the whole ring if possible
        mol = Chem.Mol(self.mol)
        atoms = list(map(mol.GetAtomWithIdx, atom_indices))
        aromatic = any(list(map(lambda atom: atom.GetIsAromatic(), atoms)))

        def get_neighbors():
            neighbors = []
            for n in list(map(lambda atom: atom.GetNeighbors(), [mol.GetAtomWithIdx(int(x)) for x in atom_indices])):
                neighbors.extend(n)
            neighbors = np.unique(list(map(lambda a: a.GetIdx(), neighbors)))
            return neighbors

        # if any aromtic atoms in neighbors, add them
        if aromatic:
            repeat = True
            while repeat:
                repeat = False
                for n in set(get_neighbors()) - set(atom_indices):
                    if n not in atom_indices and mol.GetAtomWithIdx(int(n)).GetIsAromatic():
                        atom_indices.append(n)
                        repeat = True

        return atom_indices
    
    def get_signature(self):
        self.calc_sig()
        
        # calc Mol from list of ints
        sub = None
        mol = mol_with_atom_index(self.mol)
        with Chem.RWMol(mol) as mw:
            for idx in set(list(range(self.mol.GetNumAtoms()))) - set(self.sig):
                mw.RemoveAtom(idx)
            sub = Chem.Mol(mw)
        return mw
    
    def get_smiles_signature(self):
        return Chem.MolToSmiles(self.get_signature())
    
    def get_centre(self):
        return self.cen
    
    def get_bond(self):
        return self.bond
                
            
        

class Reaction:
    def __init__(self, reactant, product, debug=False):
        self.reactant = RLMol(reactant)
        self.product = RLMol(product)
        self.debug = debug
        
    def _GetMCS(self):
        '''Get the Maximum common subsequence from reactant and product'''
        mcs = rdFMCS.FindMCS([self.reactant.mol, self.product.mol])
        return Chem.MolFromSmarts(mcs.smartsString)
    
    def display_reactant(self, atom_num=False, highlight=False):
        self.reactant.display_mol(atom_num, highlight)
            
            
    def display_product(self, atom_num=False, highlight=False):
        self.product.display_mol(atom_num, highlight)
    
    def calculate_centres_and_signatures(self):
        '''
        Calculates centres and signatures from reactants and products
        Returns None
        '''
        mcs = self._GetMCS()
        if self.debug:
            print("MCS")
            display(mol_with_atom_index(mcs))
            print("-"*100+"\n")
            print("Reactant\n")
        self.reactant.calculate_centres_and_signatures(mcs, self.debug)
        
        if self.debug:
            print("-"*100, "\nProduct\n")
        self.product.calculate_centres_and_signatures(mcs, self.debug)
        
        # if unequal signatures/centres obtained, try to fix:
        # CASE 1: More than 1 signature - try to merge them to reduce count
        rcen, pcen = self.get_centres()
        rmerge, pmerge = False, False
        if len(rcen) > 1:
            rmerge = self.reactant.try_to_merge(self.debug)
        if len(pcen) > 1:
            pmerge = self.product.try_to_merge(self.debug)
        
        # if only one merge was successful, push back the centre by one atom in the other
        if self.debug and ((rmerge and not pmerge) or (pmerge and not rmerge)):
            print("-"*100+"\n")
            print("Merge..")
        if rmerge and not pmerge:
            self.product.push_back_centre(self.reactant.get_centre_in_cs(merge=True), self.debug)
        elif pmerge and not rmerge:
            self.reactant.push_back_centre(self.product.get_centre_in_cs(merge=True), self.debug)
            
        # CASE 2: Reactant signature is Hydrogen
        rcen, pcen = self.get_centres()
        if len(rcen) == 0 and len(pcen) != 0:
            if self.debug:
                print("-"*100+"\n")
                print("Pushback..")
            self.reactant.push_back_centre(self.product.get_centre_in_cs(merge=False), self.debug)
            self.product.push_back_centre(self.reactant.get_centre_in_cs(merge=False), self.debug)
    
    def get_signatures(self):
        # calc Mol from atom indices
        return self.reactant.get_signature(), self.product.get_signature()
    
    def get_subgraphs(self):
        return self.reactant.get_subgraph(), self.product.get_subgraph()
    
    def get_smiles_subgraphs(self):
        return self.reactant.get_smiles_subgraph(), self.product.get_smiles_subgraph()
    
    def get_smiles_signatures(self):
        return self.reactant.get_smiles_signature(), self.product.get_smiles_signature()
    
    def get_centres(self):
        return self.reactant.get_centre(), self.product.get_centre()

    def get_bonds(self):
        return self.reactant.get_bond(), self.product.get_bond()
        

def sig_and_cen_collector(df, return_dict = None):
    temp_rsig_list = []
    temp_psig_list = []
    temp_rsub_list = []
    temp_psub_list = []
    temp_rcen_list = []
    temp_pcen_list = []
    temp_rbond_list = []
    temp_pbond_list = []

    for i in range(df.shape[0]):
        mol1 = Chem.MolFromSmiles(df["reactants"].iloc[i])
        mol2 = Chem.MolFromSmiles(df["products"].iloc[i])

        R = Reaction(mol1, mol2)
        R.calculate_centres_and_signatures()
        
        rcen, pcen = R.get_centres()
        rsub, psub = R.get_smiles_subgraphs()
        rsig, psig = R.get_smiles_signatures()
        rbond, pbond = R.get_bonds()
        
        temp_rsig_list.append(rsig)
        temp_psig_list.append(psig)    
        temp_rsub_list.append(rsub)
        temp_psub_list.append(psub)    
        temp_rcen_list.append(rcen)    
        temp_pcen_list.append(pcen) 
        temp_rbond_list.append(rbond)
        temp_pbond_list.append(pbond)   

    if return_dict is not None:
        return_dict["rsig"] += temp_rsig_list
        return_dict["psig"] += temp_psig_list
        return_dict["rsub"] += temp_rsub_list
        return_dict["psub"] += temp_psub_list
        return_dict["rcen"] += temp_rcen_list
        return_dict["pcen"] += temp_pcen_list
        return_dict["rbond"] += temp_rbond_list
        return_dict["pbond"] += temp_pbond_list
    else:
        return temp_rsig_list, temp_psig_list, temp_rsub_list, temp_psub_list, temp_rcen_list, temp_pcen_list, temp_rbond_list, temp_pbond_list

#######################################
# CALLING FUNCTIONS - MULTIPROCESSING #
#######################################

# multiprocess it - cuz some reactions go into infinite loops
from multiprocessing import Process, Manager
import time

def multiprocess_collector(df, return_dict):
    '''
    Collects signatures and centres for df using multiprocessing.
    Due to some reactions taking too long, the multiprocessing happens in recursion - 100 -> 10 -> 1 (size of df to process)
    Then all the results are collected and returned
    '''
    print(f"GOT DF OF SHAPE {df.shape}")
    
    for x in ["rsig", "psig", "rsub", "psub", "rcen", "pcen", 'rbond', 'pbond']:
        if x not in return_dict:
            return_dict[x] = []
    # spawn process to run on whole df
    p = Process(target=sig_and_cen_collector, args=(df, return_dict))
    p.start()
    start_time = time.time()

    # terminate if takes too long
    done = False
    kill_threshold = 10
    while not done:
        if p.is_alive():
            if time.time() - start_time > kill_threshold:
                print("Killing.....")
                p.kill()
                break
        else:
            done = True
    p.join()

    # if completed successfully, return results
    if done:
        return return_dict["rsig"], return_dict["psig"], return_dict["rsub"], return_dict["psub"], return_dict["rcen"], return_dict["pcen"], return_dict["rbond"], return_dict["pbond"]

    # not done - if df of size 1, return defaults instead
    if df.shape[0] == 1:
        return_dict["rsig"] += ['']
        return_dict["psig"] += ['']
        return_dict["rsub"] += ['']
        return_dict["psub"] += ['']
        return_dict["rcen"] += [[]]
        return_dict["pcen"] += [[]]
        return_dict["rbond"] += [[]]
        return_dict["pbond"] += [[]]
        return [''], [''], [''], [''], [[]], [[]], [[]], [[]]
    
    # not done - else divide df into 10 parts and repeat
    elements = 100
    if df.shape[0] % 10 != 0:  # add some to make divisible by 10
        elements = df.shape[0]
        while df.shape[0] % 10 != 0:
            df = pd.concat([df, df.iloc[:100-df.shape[0]]])
    
    step_size = df.shape[0] // 10
    for i in range(10):
        multiprocess_collector(df.iloc[i*step_size: (i+1)*step_size], return_dict)
    return return_dict["rsig"][:elements], return_dict["psig"][:elements], return_dict["rsub"][:elements], return_dict["psub"][:elements], \
                return_dict["rcen"][:elements], return_dict["pcen"][:elements], return_dict["rbond"][:elements], return_dict["rbond"][:elements]



if __name__ == "__main__":
    n = 100
    manager = Manager()
    rsig_list = []
    psig_list = []
    rsub_list = []
    psub_list = []
    rcen_list = []
    pcen_list = []
    rbond_list = []
    pbond_list = []

    dataset = pd.read_csv("datasets/my_uspto/processed_data.csv", index_col=0)
    for i in range(dataset.shape[0]//n+1):
        print("\n\n\n")
        print(i*n, min(i*n+n, dataset.shape[0]))
        man_dict = manager.dict()
        a, b, c, d, e, f, g, h = multiprocess_collector(dataset.iloc[i*n:min(i*n+n, dataset.shape[0])], man_dict)
        rsig_list.extend(a)
        psig_list.extend(b)
        rsub_list.extend(c)
        psub_list.extend(d)
        rcen_list.extend(e)
        pcen_list.extend(f)
        rbond_list.extend(g)
        pbond_list.extend(h)

    dataset = dataset.drop("reagents", axis=1)

    dataset["rsig"] = rsig_list
    dataset["psig"] = psig_list
    dataset["rsub"] = rsub_list
    dataset["psub"] = psub_list
    dataset["rcen"] = rcen_list
    dataset["pcen"] = pcen_list
    dataset["rbond"] = rbond_list
    dataset["pbond"] = pbond_list

    file = "datasets/my_uspto/simulator_dataset.csv"
    dataset.to_csv(file)
    print("Dumped file at", file)
    print("Final shape:", dataset.shape)
    print("Final columns:", dataset.columns)