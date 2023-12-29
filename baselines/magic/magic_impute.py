import magic
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_magic_output(dataset_name,task_name):

    input_csv = f'./outputs/{dataset_name}/{task_name}/corrupted.csv'
    #print current working directory
    #change the current working directory to the parent directory


    input_csv = Path(input_csv)



    magic_op = magic.MAGIC()

    data = pd.read_csv(input_csv, header=None)
    data_magic = magic_op.fit_transform(data)

    df = pd.DataFrame(data_magic)

    out_dir = f'./baseline_outputs/magic/{dataset_name}/{task_name}/{dataset_name}_magic_out.csv'

    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)

    #save the output as a csv file

    df.to_csv(out_dir, header=False, index=False)

if __name__ == "__main__":
    os.chdir('../../')

    names_list = ['ziesel', 'klein']
    task_list = ['zero_one_dropout','zero_two_dropout','zero_four_dropout']

    for name in names_list:
        for task in task_list:
            get_magic_output(name,task)




