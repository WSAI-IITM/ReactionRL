##############################
# Download USPTO-MIT dataset #
##############################

from deepchem.molnet import load_uspto
tasks, dataset, transformers = load_uspto(featurizer="plain")

###############################
# Convert to pandas dataframe #
###############################
from rdkit import Chem
import numpy as np
import pandas as pd

dataset = dataset[0]
dataset = list(map(lambda x: x.split(">"), dataset.X))
dataset = np.array(dataset)

dataset = pd.DataFrame({"reactants": dataset[:, 0], "reagents": dataset[:, 1], "products": dataset[:, 2]})

#######################
# Basic preprocessing #
#######################
print("Starting shape:", dataset.shape)

# first preprocess products
products = dataset["products"].tolist()
products = list(map(lambda x: x.split("."), products))

# remove small products
for i in range(len(products)):
    products[i] = list(filter(lambda x: len(x) > 6, products[i]))
dataset["products"] = products
print("Internally removed small products.", dataset.shape)

# add helper column with number of products
num_prod = list(map(len, products))
dataset["num_prod"] = num_prod
print("Added helper column with number of products.", dataset.shape)

# remove reactions that are left with no products
dataset = dataset[dataset["num_prod"] != 0]
print("Removed reactions with no products:", dataset.shape)

# divide reactions with more than 1 product
temp_df = dataset[dataset["num_prod"] != 1]
dataset = dataset[dataset["num_prod"] == 1]
dataset.reset_index(inplace=True)

for r in range(temp_df.shape[0]):
    row = temp_df.iloc[r]
    for product in row["products"]:
        new_row = row.copy()
        new_row["products"] = product
        dataset.loc[dataset.shape[0]+1] = new_row
if "index" in dataset.columns:
    dataset.drop(columns=["index"], inplace=True)
print("Divided reactions with multiple products :", dataset.shape)

# remove helper column
dataset.drop(columns=["num_prod"], inplace=True)
print("Removed helper column", dataset.shape)

# Some are still lists, make them strings again
dataset["products"] = dataset["products"].apply(lambda x: x[0] if isinstance(x, list) else x)

#####################################
# Extract molecular transformations #
#####################################

from rdkit.Chem.Descriptors import ExactMolWt

def mol_weight(smile):
    return ExactMolWt(Chem.MolFromSmiles(smile))

def matching_reactant(reactants, product):
    '''
    Takes a smile string of reactants and a single product.
    Returns the reactant which has the closest molecular weight to the product
    '''
    reactants = reactants.split(".")
    rmw = np.array(list(map(mol_weight, reactants)))
    pmw = mol_weight(product)
    return reactants[np.abs(rmw-pmw).argmin()]
    

dataset["reactants"] = dataset.apply(lambda row: matching_reactant(row["reactants"], row["products"]), axis=1)
print("Dataset after 1 to 1 mapping completed", dataset.shape)

# Remove duplicates
dataset = dataset.drop_duplicates(subset=set(["reactants", "products"]), keep="first", ignore_index=True)
print("After removing duplicates:", dataset.shape)

# remove reactant = product
dataset = dataset.loc[(dataset["reactants"] != dataset["products"])]
print("After removing reactant = product", dataset.shape)

# Save
dataset.to_csv("datasets/my_uspto/processed_data.csv")