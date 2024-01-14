import numpy as np
import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tqdm import tqdm


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(file_path1, file_path2):
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(file_path1, header=None)
    df2 = pd.read_csv(file_path2, header=None)
    pc1 = PCA(n_components=2)
    pc2 = PCA(n_components=2)

    #transpose the dataframes
    df1 = df1.T
    df2 = df2.T

    #reduce the dimentions of the dataframes
    df1 = pc1.fit_transform(df1)
    df2 = pc2.fit_transform(df2)

    #claculate rmse
    rmse = np.sqrt(np.mean(np.square(df1 - df2)))

    #reduce dimenttions of the dataframes using PCA



    # Aggregate the similarity scores to get an overall similarity measure
    overall_similarity = rmse

    return overall_similarity



if __name__ == '__main__':

    os.chdir('../')

    train_datasets_1 = ["klein","ziesel",
                         'muraro', 'quake_10x_bladder', 'romanov',
                         ]

    train_datasets_2 = ["young",
                        "tosches turtle",
                        "quake_10x_spleen",
                        "quake_smart-seq2_heart",
                        "quake_smart-seq2_lung"]


    tr1_dict = {i:[] for i in train_datasets_1}

    for dataset in tqdm(train_datasets_1):
        for d in tqdm(train_datasets_2):

            csv_path_td1 = f'data/split_data/train/{dataset}/original/{dataset}.csv'
            csv_path_td2 = f'data/split_data/train/{d}/original/{d}.csv'

            sim = calculate_similarity(csv_path_td1, csv_path_td2)

            tr1_dict[dataset].append(sim)


    #in the dictionary replace list with mean of the list

    for key in tr1_dict.keys():
        tr1_dict[key] = np.mean(tr1_dict[key])

    #plot barplot

    plt.bar(tr1_dict.keys(), tr1_dict.values())
    plt.xlabel("Dataset")
    plt.ylabel("Mean RMSE")
    plt.show()


