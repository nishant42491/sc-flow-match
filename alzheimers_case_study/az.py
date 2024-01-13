import scanpy as sc
import pandas as pd
import os
import pathlib
import anndata
import numpy as np
import random


#set seeds

np.random.seed(42)
random.seed(42)


def make_az_data(root_dir, metadata_file):

    #open the tsv metadata file and read it into a dataframe

    #go through each file in the root directory and if .mtx file is found follow the stuff below to make datasets and append it below one another:
    ad_lists = []
    for f in os.listdir(root_dir):

        if f.endswith('.mtx'):
            metadata = pd.read_csv(metadata_file, sep='\t', index_col=0)
            #read the mtx file
            ad = sc.read_mtx(os.path.join(root_dir,f))
            ad = ad.T
            p = pathlib.Path(os.path.join(root_dir,f))
            fname = p.stem
            #replace _matrix.mtx with _genes.tsv
            fname = fname.replace('_matrix.mtx','')
            fname = fname.replace('_matrix','')
            gens = pd.read_csv(os.path.join(root_dir,fname+'_genes.tsv'), sep='\t', header=None)
            bcode = pd.read_csv(os.path.join(root_dir,fname+'_barcodes.tsv'), sep='\t', header=None)
            #replace the -1 in all barcodes with _fname
            bcode[0] = bcode[0].str.replace('-1',f'_{fname}')
            #fimd teh intersection of metadata nd bcode

            ad.obs_names = bcode[0]
            ad.var_names = gens[0]

            idx = bcode[0]
            #extract the batchCond column from the metadata file where its index matches bcode
            idx = metadata.index.isin(idx)

            #filter the metadata based on the intersection
            metadata = metadata[idx]
            labs = metadata["batchCond"]
            cells = metadata["cellType"]
            #save cell types in anndata
            ad.obs['cellType'] = cells

            #save labels in anndata

            ad.obs['labels'] = labs

            ad_lists.append(ad)

        #concatenate all the anndatas in ad_lists
    ad = anndata.concat(ad_lists, axis=0)

    x = ad.obs_names.values
    #shuffle the rows
    np.random.shuffle(x)
    #shuffle ad
    ad = ad[x]

    #remove all the rows ehre the obs labels or obs celltype is nan
    ad = ad[~ad.obs['labels'].isnull()]
    ad = ad[~ad.obs['cellType'].isnull()]

    return ad


def preprocess_alheimers(root_dir, metadata_dir):
    # Read the h5ad file
    adata = make_az_data(root_dir, metadata_dir)

    # Data filtering and quality control
    sc.pp.filter_genes(adata, min_cells=int(0.01 * adata.n_obs))
    sc.pp.filter_cells(adata, min_genes=int(0.01 * adata.n_vars))

    # Normalization by size factor and log transformation
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Selecting highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2048)

    # Getting the indices of the top highly variable genes
    top_genes_indices = np.where(adata.var['highly_variable'])[0]

    # Selecting the top highly variable genes for training
    adata_processed = adata[:, top_genes_indices].copy()

    #split into 80/20 train test and visualise umap of test set

    ad_train, ad_test = split_data(adata_processed, split=0.8)

    train_X = ad_train.X
    #from csr np matrix to dense np matrix
    train_X = train_X.todense()
    train_cell_names = ad_train.obs_names.values
    train_gene_names = ad_train.var_names.values
    train_labels = ad_train.obs['labels'].values
    #save as string
    train_labels = train_labels.astype(str)
    train_cell_types = ad_train.obs['cellType'].values


    test_X = ad_test.X
    #from csr np matrix to dense np matrix
    test_X = test_X.todense()
    test_cell_names = ad_test.obs_names.values
    #save as string
    test_cell_names = test_cell_names.astype(str)
    test_gene_names = ad_test.var_names.values
    test_gene_names = test_gene_names.astype(str)
    test_labels = ad_test.obs['labels'].values
    #save as string
    test_labels = test_labels.astype(str)
    test_cell_types = ad_test.obs['cellType'].values
    test_cell_types = test_cell_types.astype(str)

    #perform 40% dropout on the test set and train set

    dropout_value = 0.4

    train_mask = np.random.random(train_X.shape) < dropout_value
    test_mask = np.random.random(test_X.shape) < dropout_value

    #make a copy of the train and test sets and apply masks

    train_X_40 = train_X.copy()
    test_X_40 = test_X.copy()

    train_X_40[train_mask] = 0
    test_X_40[test_mask] = 0

    #perform 20% dropout on the test set and train set

    dropout_value = 0.2
    train_mask = np.random.random(train_X.shape) < dropout_value
    test_mask = np.random.random(test_X.shape) < dropout_value

    #make a copy of the train and test sets and apply masks

    train_X_20 = train_X.copy()
    test_X_20 = test_X.copy()

    train_X_20[train_mask] = 0
    test_X_20[test_mask] = 0

    #perform 10% dropout on the test set and train set

    dropout_value = 0.1
    train_mask = np.random.random(train_X.shape) < dropout_value
    test_mask = np.random.random(test_X.shape) < dropout_value

    #make a copy of the train and test sets and apply masks

    train_X_10 = train_X.copy()
    test_X_10 = test_X.copy()

    train_X_10[train_mask] = 0
    test_X_10[test_mask] = 0

    og_train_path = rf'data/split_data/train/alzheimer/original'
    og_test_path = rf'data/split_data/test/alzheimer/original'

    pathlib.Path(og_train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(og_test_path).mkdir(parents=True, exist_ok=True)

    #save numpy arrays as csv files
    np.savetxt(rf'{og_train_path}/alzheimer.csv', train_X, delimiter=',')
    np.savetxt(rf'{og_test_path}/alzheimer.csv', test_X, delimiter=',')

    #Save lalbels cell names gene names and cell types inside the test and train alzheimers folders as csv files


    np.savetxt(rf'{og_train_path}/labels/alzheimer_labels.csv', train_labels, delimiter=',', fmt='%s')
    np.savetxt(rf'{og_test_path}/labels/alzheimer_labels.csv', test_labels, delimiter=',', fmt='%s')
    np.savetxt(rf'{og_train_path}/alzheimer_cell_names.csv', train_cell_names, delimiter=',', fmt='%s')
    np.savetxt(rf'{og_test_path}/alzheimer_cell_names.csv', test_cell_names, delimiter=',', fmt='%s')
    np.savetxt(rf'{og_train_path}/alzheimer_gene_names.csv', train_gene_names, delimiter=',', fmt='%s')
    np.savetxt(rf'{og_test_path}/alzheimer_gene_names.csv', test_gene_names, delimiter=',', fmt='%s')
    np.savetxt(rf'{og_train_path}/alzheimer_cell_types.csv', train_cell_types, delimiter=',', fmt='%s')
    np.savetxt(rf'{og_test_path}/alzheimer_cell_types.csv', test_cell_types, delimiter=',', fmt='%s')



    #save the 40%  20% and 10% dropout numpy arrays as csv files

    dropout_40_train_path = rf'data/split_data/train/alzheimer/zero_four_dropout'
    dropout_40_test_path = rf'data/split_data/test/alzheimer/zero_four_dropout'

    dropout_20_train_path = rf'data/split_data/train/alzheimer/zero_two_dropout'
    dropout_20_test_path = rf'data/split_data/test/alzheimer/zero_two_dropout'

    dropout_10_train_path = rf'data/split_data/train/alzheimer/zero_one_dropout'
    dropout_10_test_path = rf'data/split_data/test/alzheimer/zero_one_dropout'

    pathlib.Path(dropout_40_train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dropout_40_test_path).mkdir(parents=True, exist_ok=True)

    pathlib.Path(dropout_20_train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dropout_20_test_path).mkdir(parents=True, exist_ok=True)

    pathlib.Path(dropout_10_train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dropout_10_test_path).mkdir(parents=True, exist_ok=True)

    np.savetxt(rf'{dropout_40_train_path}/alzheimer.csv', train_X_40, delimiter=',')
    np.savetxt(rf'{dropout_40_test_path}/alzheimer.csv', test_X_40, delimiter=',')

    np.savetxt(rf'{dropout_20_train_path}/alzheimer.csv', train_X_20, delimiter=',')
    np.savetxt(rf'{dropout_20_test_path}/alzheimer.csv', test_X_20, delimiter=',')

    np.savetxt(rf'{dropout_10_train_path}/alzheimer.csv', train_X_10, delimiter=',')
    np.savetxt(rf'{dropout_10_test_path}/alzheimer.csv', test_X_10, delimiter=',')


    return None

def split_data(ad, split=0.8):

    #shuffle the rows

    X = ad.X
    lenx= X.shape[0]

    cell_names = ad.obs_names.values
    gene_names = ad.var_names.values

    labels = ad.obs['labels'].values
    cell_types = ad.obs['cellType'].values

    row_idices = np.arange(lenx)
    np.random.shuffle(row_idices)

    X = X[row_idices]
    cell_names = cell_names[row_idices]
    labels = labels[row_idices]
    cell_types = cell_types[row_idices]

    ad = anndata.AnnData(X=X, obs=pd.DataFrame({'labels':labels, 'cellType':cell_types}, index=cell_names), var=pd.DataFrame(index=gene_names))

    split = int(split*ad.shape[0])

    ad_train = ad[:split]
    ad_test = ad[split:]

    return ad_train, ad_test




if __name__ == '__main__':

    os.chdir('../')

    preprocess_alheimers(r'data/AZ/mtx', r'data/AZ/scRNA_metadata.tsv')



