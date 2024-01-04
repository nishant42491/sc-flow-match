# %%
import numpy as np
import scanpy as sc
from scipy import sparse
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from GNNImpute.api import GNNImpute
from anndata import AnnData
import pathlib

# %%


def calculate_GNNimp_score(dataset_name, task_name):

    adata_train = pd.read_csv(rf'D:\Nishant\cfm_impute\flow-matching-single cell\data\split_data\train\{dataset_name}\{task_name}\{dataset_name}.csv', header=None)
    adata_train = AnnData(adata_train.values)

    adata_test = pd.read_csv(fr'D:\Nishant\cfm_impute\flow-matching-single cell\data\split_data\test\{dataset_name}\{task_name}\{dataset_name}.csv', header=None)
    adata_test = AnnData(adata_test)

    og_anndata_train = pd.read_csv(fr'D:\Nishant\cfm_impute\flow-matching-single cell\data\split_data\train\{dataset_name}\original\{dataset_name}.csv', header=None)
    og_anndata_train = AnnData(og_anndata_train.values)
    og_anndata_test = pd.read_csv(fr'D:\Nishant\cfm_impute\flow-matching-single cell\data\split_data\test\{dataset_name}\original\{dataset_name}.csv', header=None)
    og_anndata_test = AnnData(og_anndata_test.values)
    #Find a way to attach the og stuff to adata stuff
    adata_train.raw = og_anndata_train
    adata_test.raw = og_anndata_test




    # %%

    adata = GNNImpute(adata_train=adata_train,
                      adata_val=adata_test,
                      layer='GCNConv',
                      no_cuda=False,
                      epochs=1500,
                      lr=0.001,
                      weight_decay=0.0005,
                      hidden=50,
                      patience=200,
                      fastmode=False,
                      heads=1,
                      use_raw=False,
                      verbose=True)

    # %%

    output_dir = fr"D:\Nishant\cfm_impute\flow-matching-single cell\baseline_outputs\gnnimpute\{dataset_name}\{task_name}\gnnimpute_out.csv"
    pathlib.Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    dropout_pred = adata.X[adata.obs.idx_test]
    #save the dropout prediction
    np.savetxt(output_dir, dropout_pred, delimiter=",")

if __name__ == "__main__":

    datasets = ['muraro', 'plasschaert', 'romanov', 'tosches turtle',
                "young", "quake_10x_bladder", "quake_10x_limb_muscle", "quake_10x_spleen",
                "quake_smart-seq2_diaphragm", "quake_smart-seq2_heart", "quake_smart-seq2_limb_muscle",
                "quake_smart-seq2_lung", "quake_smart-seq2_trachea", "klein", "ziesel"]

    tasks = ['zero_one_dropout', 'zero_two_dropout', 'zero_four_dropout']

    for dataset in datasets:
        for task in tasks:
            calculate_GNNimp_score(dataset, task)


