import scanpy as sc
import numpy as np
import os



if __name__ == '__main__':

    os.chdir('../')



    klein_data = sc.read_csv("data/split_data/test/klein/zero_one_dropout/klein.csv")
    klein_labels = np.squeeze(np.loadtxt("data/split_data/test/klein/original/labels/klein_labels.csv", delimiter=","))
    #save labels to anndata object
    klein_data.obs['labels'] = klein_labels

    output_klein = sc.read_csv("outputs/klein/zero_one_dropout/gen_out.csv")
    output_klein_labels = np.squeeze(np.loadtxt("data/split_data/test/klein/original/labels/klein_labels.csv", delimiter=","))
    output_klein.obs['labels'] = output_klein_labels
    #perform diffusion psuedotime analysis

    sc.pp.neighbors(klein_data)
    sc.tl.diffmap(klein_data)
    sc.tl.dpt(klein_data, n_branchings=1, n_dcs=10)
    sc.pl.diffmap(klein_data, color=['labels'])
