'''
To run: time python -m reactionrl.preprocessing.filter_action_dataset 2> errors.txt

This file marks those actions which give error in the database.
'''

import time
from rdkit import Chem
import pandas as pd
from reactionrl.actions import apply_action, get_applicable_actions
from reactionrl.actions.action_space import get_default_action_space
from multiprocessing import Process, Manager

dataset = pd.read_csv("datasets/my_uspto/action_dataset.csv", index_col=0)

# Reset the index to be sequential
dataset.reset_index(inplace=True)
dataset = dataset.drop(columns=["index"])


# Init/Reset these columns
dataset["reactant_works"] = [True]*dataset.shape[0]
dataset["reactant_tested"] = [False]*dataset.shape[0]

dataset["action_works"] = [True] * dataset.shape[0]
dataset["action_tested"] = [False] * dataset.shape[0]

def find_bad_action_index(in_mol, temp_df, return_q=None):
    bad_actions = []
    for j in range(temp_df.shape[0]):
        action = temp_df.iloc[j]

        # Try to apply action/
        try:
            apply_action(in_mol, action["rsub"], action["rcen"], action["rsig"], action["rsig_cs_indices"],
                                    action["psub"], action["pcen"], action["psig"], action["psig_cs_indices"])
        except Exception as e:
            bad_actions.append(action.name)
    if return_q:
        return_q.put(bad_actions)
    else:
        return bad_actions


if __name__ == "__main__":
    count = 0
    process_limit = 11
    t = time.time()
    process_list = []
    manager = Manager()
    result_q = manager.Queue()

    action_space = get_default_action_space()

    for i in range(0, dataset.shape[0]):
        in_mol = Chem.MolFromSmiles(dataset.iloc[i]["reactants"])
        dataset["reactant_tested"].iat[i] = True

        # Try out all the actions
        temp_df = dataset[dataset["rsig_clusters"].isin(action_space.get_applicable_rsig_clusters(in_mol))]
        if temp_df.shape[0] == 0:
            dataset["reactant_works"].iat[i] = False
        else:
            # Tested's
            dataset.loc[temp_df.index, "action_tested"] = True

            # Start a process to find bad actions
            process_list.append(Process(target=find_bad_action_index, args=(in_mol, temp_df, result_q)))
            process_list[-1].start()

            # Completed processes
            if len(process_list) == process_limit:
                p = process_list.pop(0)
                p.join()
                bad_actions = result_q.get()
                for action in bad_actions:
                    dataset["action_works"].at[action] = False
            print(i, time.time()-t, f"{dataset['reactant_tested'].sum()}({dataset.loc[dataset['reactant_tested']]['reactant_works'].sum()})",
                                        f"{dataset['action_tested'].sum()}({dataset.loc[dataset['action_tested']]['action_works'].sum()})")
        t = time.time()

    # Get the remaining processes
    while process_list:
        p = process_list.pop(0)
        p.join()
        bad_actions = result_q.get()
        for action in bad_actions:
            dataset["action_works"].at[action] = False
        print(i, time.time()-t, f"{dataset['reactant_tested'].sum()}({dataset.loc[dataset['reactant_tested']]['reactant_works'].sum()})",
                                        f"{dataset['action_tested'].sum()}({dataset.loc[dataset['action_tested']]['action_works'].sum()})")

    print("Dumping...")
    dataset.to_csv("datasets/my_uspto/action_dataset-filtered.csv")
