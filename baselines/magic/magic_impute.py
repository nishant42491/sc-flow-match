import magic
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_magic_output(dataset_name,task_name):

    input_csv = f'./outputs/{dataset_name}/{task_name}/corrupted.csv'
    train_main = fr'data/split_data/train/{dataset_name}/original/{dataset_name}.csv'
    #print current working directory
    #change the current working directory to the parent directory

    train_array = pd.read_csv(train_main, header=None)


    input_csv = Path(input_csv)

    magic_op = magic.MAGIC()

    data = pd.read_csv(input_csv, header=None)
    lendata = len(data)
    lentrain = len(train_array)

    #concatenate the train and test data
    data = pd.concat([train_array, data], axis=0)

    data_magic = magic_op.fit_transform(data)

    df = pd.DataFrame(data_magic)

    df = df.iloc[lentrain:,:]

    out_dir = f'./baseline_outputs/magic/{dataset_name}/{task_name}/{dataset_name}_magic_out.csv'

    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)

    #save the output as a csv file

    df.to_csv(out_dir, header=False, index=False)

if __name__ == "__main__":
    os.chdir('../../')

    '''names_list = ['muraro','plasschaert','romanov','tosches turtle',
                  "young", "quake_10x_bladder","quake_10x_limb_muscle", "quake_10x_spleen",
                  "quake_smart-seq2_diaphragm", "quake_smart-seq2_heart", "quake_smart-seq2_limb_muscle",
                  "quake_smart-seq2_lung", "quake_smart-seq2_trachea", "klein", "ziesel"]'''

    names_list = ['alzheimer']

    #names_list = ['klein','ziesel']
    task_list = ['zero_one_dropout','zero_two_dropout','zero_four_dropout']

    for name in names_list:
        for task in task_list:
            get_magic_output(name,task)




