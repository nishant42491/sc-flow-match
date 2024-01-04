import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import median_absolute_error
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from pathlib import Path
import numpy as np



def cos_sim(a, b):

    z = np.dot(a, b)
    x = np.linalg.norm(a)
    y = np.linalg.norm(b)
    return z/(x*y)

def compute_scores(original_csv, resultant_csv,dataset_name, dropout_eval_name):
    original_df = pd.read_csv(original_csv, header=None)
    resultant_df = pd.read_csv(resultant_csv, header=None)

    rmse = np.sqrt(mean_squared_error(original_df.values, resultant_df.values))
    cos_list = []

    for i,j in zip(original_df.values, resultant_df.values):
        cos_list.append(cos_sim(i,j))

    cos_similarity = np.mean(np.array(cos_list))
    # calculate the pearsons correlation coefficient

    # computer the mean L1 distance
    mean_l1_error = np.mean(np.abs(original_df.values - resultant_df.values))


    # Prepare the results in a log string
    log = f"Dataset: {dataset_name}\n"
    log += f"RMSE: {rmse}\n"
    log += f"Cosine_Similarity: {cos_similarity}\n"
    log += f"L1_Distance: {mean_l1_error}\n"

    save_dir = f'baseline_eval/gnnimpute/{dataset_name}/{dropout_eval_name}/{dataset_name}_results_log.txt'
    Path(save_dir).parent.mkdir(parents=True, exist_ok=True)
    # Write results to a text file
    with open(save_dir, "w") as file:
        file.write(log)


def compute_cluster_scores(original_csv, resultant_csv,dataset_name, dropout_eval_name):

    original_df = pd.read_csv(original_csv, header=None)
    resultant_df = pd.read_csv(resultant_csv, header=None)

    test_labels = pd.read_csv(f'./data/split_data/test/{dataset_name}/original/labels/{dataset_name}_labels.csv', header=None)

    num_unique_labels = len(np.unique(test_labels.values))

    kmeans = KMeans(n_clusters=num_unique_labels, random_state=42).fit(original_df.values)
    original_labels = kmeans.labels_

    resultant_labels = kmeans.predict(resultant_df.values)

    ari = adjusted_rand_score(original_labels, resultant_labels)
    nmi = normalized_mutual_info_score(original_labels, resultant_labels)

    # Prepare the results in a log string
    log = f"Dataset: {dataset_name}\n"
    log += f"ARI: {ari}\n"
    log += f"NMI: {nmi}\n"

    save_dir = f'baseline_eval/gnnimpute/{dataset_name}/{dropout_eval_name}/{dataset_name}_cluster_results_log.txt'
    Path(save_dir).parent.mkdir(parents=True, exist_ok=True)
    # Write results to a text file
    with open(save_dir, "w") as file:
        file.write(log)


if __name__ == "__main__":

    os.chdir('../../')

    dataset_name = ['ziesel', 'klein']
    dropout_eval_name = ['zero_four_dropout', 'zero_two_dropout', 'zero_one_dropout']

    for dataset in dataset_name:
        for dropout_eval in dropout_eval_name:
            og_csv = f'outputs/{dataset}/{dropout_eval}/og_out.csv'
            gen_out = f'baseline_outputs/gnnimpute/{dataset}/{dropout_eval}/gnnimpute_out.csv'
            compute_scores(og_csv, gen_out, dataset, dropout_eval)

    dataset_name = ['muraro', 'plasschaert', 'romanov', 'tosches turtle',
                    "young", "quake_10x_bladder", "quake_10x_limb_muscle", "quake_10x_spleen",
                    "quake_smart-seq2_diaphragm", "quake_smart-seq2_heart", "quake_smart-seq2_limb_muscle",
                    "quake_smart-seq2_lung", "quake_smart-seq2_trachea"]

    dropout_eval_name = ['zero_four_dropout', 'zero_two_dropout', 'zero_one_dropout']

    for dataset in dataset_name:
        for dropout_eval in dropout_eval_name:
            og_csv = f'outputs/{dataset}/{dropout_eval}/og_out.csv'
            gen_out = f'baseline_outputs/gnnimpute/{dataset}/{dropout_eval}/gnnimpute_out.csv'
            # compute_scores(og_csv, gen_out, dataset, dropout_eval)
            compute_cluster_scores(og_csv, gen_out, dataset, dropout_eval)
