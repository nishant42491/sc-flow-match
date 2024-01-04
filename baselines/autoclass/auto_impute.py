import pandas as pd
import numpy as np
from pathlib import Path
from AutoClass.AutoClass import AutoClassImpute

def get_autoclass_output(dataset_name, task_name, labels:bool = False):

    if dataset_name == "klein" or dataset_name == "ziesel":
        labels = False

    else:
        labels = True

    input_csv = fr'D:\Nishant\cfm_impute\flow-matching-single cell\data\split_data\test\{dataset_name}\{task_name}\{dataset_name}.csv'

    if labels:
        label_csv = fr'D:\Nishant\cfm_impute\flow-matching-single cell\data\split_data\test\{dataset_name}\original\labels\{dataset_name}_labels.csv'
        labs = pd.read_csv(label_csv, header=None).values
        labs = labs.astype(int)
        #find number of unique labels
        num_labs = len(np.unique(labs))
    else:
        num_labs = None

    d = pd.read_csv(input_csv, header=None).values

    if num_labs:
        ac = AutoClassImpute(d, num_cluster = [num_labs - 1, num_labs, num_labs + 1],
                             log1p=False, cellwise_norm=False)

    else:
        ac = AutoClassImpute(d,log1p=False, cellwise_norm=False)


    ac = ac['imp']
    print("pass")



    out_dir = rf'D:\Nishant\cfm_impute\flow-matching-single cell/baseline_outputs/autoclass/{dataset_name}/{task_name}/autoclass_imputed.csv'

    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir, ac, delimiter=",")

if __name__ == "__main__":

    datasets = ['muraro','plasschaert','romanov','tosches turtle',
                    "young", "quake_10x_bladder","quake_10x_limb_muscle", "quake_10x_spleen",
                    "quake_smart-seq2_diaphragm", "quake_smart-seq2_heart", "quake_smart-seq2_limb_muscle",
                    "quake_smart-seq2_lung", "quake_smart-seq2_trachea", "klein", "ziesel"]
    tasks = ['zero_one_dropout','zero_two_dropout','zero_four_dropout']

    for dataset in datasets:
        for task in tasks:
            get_autoclass_output(dataset,task)
