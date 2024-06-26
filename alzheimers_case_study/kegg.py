import scanpy
import pandas as pd
import numpy as np
import os
from anndata import concat
import pathlib
import anndata
import scanpy as sc
import gseapy as gp
from gseapy import barplot, dotplot
import matplotlib.pyplot as plt


def make_anndata(task_name, test, original):

    if not test:
        if original:
            alz_path = rf'data/split_data/train/alzheimer/{task_name}/alzheimer.csv'


        else:
            alz_path = rf'outputs/alzheimer/{task_name}/gen_out.csv'

        cell_names_path = rf'data/split_data/train/alzheimer/original/alzheimer_cell_names.csv'
        cell_types_path = rf'data/split_data/train/alzheimer/original/alzheimer_cell_types.csv'
        gene_names_path = rf'data/split_data/train/alzheimer/original/alzheimer_gene_names.csv'
        labels_path = rf'data/split_data/train/alzheimer/original/labels/alzheimer_labels.csv'

    else:
        if original:
            alz_path = rf'data/split_data/test/alzheimer/{task_name}/alzheimer.csv'

        else:
            alz_path = rf'outputs/alzheimer/{task_name}/gen_out.csv'
        #alz_path = rf'data/split_data/test/alzheimer/{task_name}/alzheimer.csv'
        cell_names_path = rf'data/split_data/test/alzheimer/original/alzheimer_cell_names.csv'
        cell_types_path = rf'data/split_data/test/alzheimer/original/alzheimer_cell_types.csv'
        gene_names_path = rf'data/split_data/test/alzheimer/original/alzheimer_gene_names.csv'
        labels_path = rf'data/split_data/test/alzheimer/original/labels/alzheimer_labels.csv'


    alz_data = pd.read_csv(alz_path, header=None).values
    cell_names = pd.read_csv(cell_names_path, header=None).values
    cell_types = pd.read_csv(cell_types_path, header=None).values
    gene_names = pd.read_csv(gene_names_path, header=None).values

    labels = pd.read_csv(labels_path, header=None).values

    #squeeze the shape of the labels and cell types
    labels = np.squeeze(labels)
    cell_types = np.squeeze(cell_types)

    #create anndata object
    ad = anndata.AnnData(X=alz_data, obs=pd.DataFrame({'labels':labels, 'cell_type':cell_types}, index=cell_names), var=pd.DataFrame(index=gene_names))

    return ad


def construct_gene_id_to_name_mappings(root_dir):

    l = []
    for f in os.listdir(root_dir):
        if 'genes' in f:
            df = pd.read_csv(os.path.join(root_dir, f), header=None, sep='\t')
            l.append(df)

    df = pd.concat(l, axis=0)

    #make the first column the index of df
    df.set_index(0, inplace=True)

    return df


def load_baseline_anndata(task_name, test, dataset_name, baseline_name):

    if not test:
        alz_path = rf'baseline_outputs/{baseline_name}/{dataset_name}/{task_name}/{dataset_name}_imputed.csv'
        cell_names_path = rf'data/split_data/train/{dataset_name}/original/{dataset_name}_cell_names.csv'
        cell_types_path = rf'data/split_data/train/{dataset_name}/original/{dataset_name}_cell_types.csv'
        gene_names_path = rf'data/split_data/train/{dataset_name}/original/{dataset_name}_gene_names.csv'
        labels_path = rf'data/split_data/train/{dataset_name}/original/labels/{dataset_name}_labels.csv'

    else:
        alz_path = rf'outputs/{dataset_name}/{task_name}/gen_out.csv'
        #alz_path = rf'data/split_data/test/alzheimer/{task_name}/alzheimer.csv'
        cell_names_path = rf'data/split_data/test/{dataset_name}/original/{dataset_name}_cell_names.csv'
        cell_types_path = rf'data/split_data/test/{dataset_name}/original/{dataset_name}_cell_types.csv'
        gene_names_path = rf'data/split_data/test/{dataset_name}/original/{dataset_name}_gene_names.csv'
        labels_path = rf'data/split_data/test/{dataset_name}/original/labels/{dataset_name}_labels.csv'


    alz_data = pd.read_csv(alz_path, header=None).values
    cell_names = pd.read_csv(cell_names_path, header=None).values
    cell_types = pd.read_csv(cell_types_path, header=None).values
    gene_names = pd.read_csv(gene_names_path, header=None).values
    labels = pd.read_csv(labels_path, header=None).values

    #squeeze the shape of the labels and cell types
    labels = np.squeeze(labels)
    cell_types = np.squeeze(cell_types)

    #create anndata object
    ad = anndata.AnnData(X=alz_data, obs=pd.DataFrame({'labels':labels, 'cell_type':cell_types}, index=cell_names), var=pd.DataFrame(index=gene_names))

    return ad



if __name__ == '__main__':

    mk_genes = ['CPSF3L',
                'ACAP3',
                'TXLNA',
                'C1orf123',
                'DHCR24',
                'NADK',
                'LDLRAP1',
                'SH3BGRL3',
                'ZDHHC18',
                'LINC01358',
                'CHD5',
                'PLOD1',
                'ADGRB2',
                "MAP7D1",
                "TMEM125",
                "FUBP1",
                "SLC22A15",
                "PADI2",
                "C1QA"]

    os.chdir('../')


    ad_og_test = make_anndata('original', True, True)

    maps = construct_gene_id_to_name_mappings('data/AZ/mtx')
    cur_var_names = ad_og_test.var_names.values
    cur_var_names = [cur_var_names[i][2:-3] for i in range(len(cur_var_names))]
    #get the gene names from the gene id
    cur_var_names = maps.loc[cur_var_names]
    #drop duplicate gene names
    cur_var_names = cur_var_names[~cur_var_names.index.duplicated(keep='first')]
    ad_og_test.var_names = cur_var_names.values.flatten()

    ad_zero_four_test = make_anndata('zero_one_dropout', True, False)
    #ad_zero_four_test = load_baseline_anndata('zero_four_dropout', True, 'alzheimer', 'alra')
    ad_zero_four_test.var_names = cur_var_names.values.flatten()

    #0-1 normalise both the anndata objects

    scanpy.tl.rank_genes_groups(ad_og_test, groupby='cell_type', method='wilcoxon')
    ad_zero_four_test.uns = ad_og_test.uns


    '''sc.pl.dotplot(ad_og_test, mk_genes, groupby="cell_type", vmin=-3, vmax=3,
                  save='alz_og_dropout.png')
    sc.pl.dotplot(ad_zero_four_test, mk_genes, groupby="cell_type",
                  vmin=-3, vmax=3, save='alz_zero_four_dropout.png')

    sc.pl.rank_genes_groups_heatmap(ad_og_test, n_genes=4, swap_axes=True,
                                    vmin=-3, vmax=3, save='alz_og_heatmap.png')
    sc.pl.rank_genes_groups_heatmap(ad_zero_four_test, n_genes=4, swap_axes=True,
                                    vmin=-3, vmax=3, save='alz_zero_four_heatmap.png')
    '''

    #plot umap plots for both the anndata objects

    '''sc.pp.neighbors(ad_og_test, n_neighbors=20, n_pcs=10)
    sc.tl.umap(ad_og_test)
    sc.pl.umap(ad_og_test, color=['cell_type', 'labels'], save='alz_og_umap.png')

    sc.pp.neighbors(ad_zero_four_test, n_neighbors=20, n_pcs=10)
    sc.tl.umap(ad_zero_four_test)
    sc.pl.umap(ad_zero_four_test, color=['cell_type', 'labels'], save='alz_zero_four_umap.png')'''
