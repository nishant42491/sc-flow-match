from .model import GNNImpute as Model
from .train import train
from .utils import adata2gdata, train_val_split, normalize

import numpy as np


def GNNImpute(adata_train,
              adata_val,
              layer='GCNConv',
              no_cuda=False,
              epochs=3000,
              lr=0.001,
              weight_decay=0.0005,
              hidden=50,
              patience=200,
              fastmode=False,
              heads=3,
              use_raw=True,
              verbose=True):
    input_dim = adata_train.n_vars

    model = Model(input_dim=input_dim, h_dim=512, z_dim=hidden, layerType=layer, heads=heads)

    #adata = normalize(adata, filter_min_counts=False)
    #adata = train_val_split(adata)
    #concat adata_train and adata_val and keep track if indexin
    # tmp = np.zeros(cell_nums, dtype=bool)
    '''#tmp[idx_train] = True
    adata.obs['idx_train'] = tmp
    tmp = np.zeros(cell_nums, dtype=bool)
    tmp[idx_val] = True
    adata.obs['idx_val'] = tmp
    tmp = np.zeros(cell_nums, dtype=bool)
    tmp[idx_test] = True
    adata.obs['idx_test'] = tmp'''

    len_train = len(adata_train)
    len_val = len(adata_val)

    adata = adata_train.concatenate(adata_val)

    tmp = np.zeros(len_train+len_val, dtype=bool)
    tmp[:len_train] = True
    adata.obs['idx_train'] = tmp
    tmp = np.zeros(len_train+len_val, dtype=bool)
    tmp[len_train:] = True
    adata.obs['idx_val'] = tmp
    adata.obs['idx_test'] = tmp
    adata.obs['size_factors'] = 1.0

    gdata = adata2gdata(adata, use_raw=use_raw)

    train(gdata=gdata, model=model, no_cuda=no_cuda, epochs=epochs, lr=lr, weight_decay=weight_decay,
          patience=patience, fastmode=fastmode, verbose=verbose)

    pred = model(gdata['x'], gdata['adj'], gdata['size_factors'])

    adata.X = pred.detach().cpu()

    return adata
