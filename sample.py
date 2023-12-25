import math
import os
import time
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from src.data.single_cell_datamodule import SingleCellDataModule

from src.models.cfm_module import CFMLitModule
from src.models.components.transformer_encoder import TransformerAutoencoder

ckpt_path = r"logs/train/runs/2023-12-25_05-15-31/checkpoints/epoch_3581.ckpt"


model = CFMLitModule.load_from_checkpoint(ckpt_path,
                                          net=TransformerAutoencoder(input_dim=2048,
                                                                     embed_dim=2048,
                                                                     time_dim=512,
                                                                     num_layers=1,
                                                                     n_heads=1,
                                                                     output_dim=2048,
                                                                     dropout=0.1))
dm = SingleCellDataModule(data_dir=r"D:\Nishant\cfm_impute\flow-matching-single cell\data",
                          batch_size=64,
                          name="klein",
                          task="zero_four_dropout")
test_dl = dm.test_dataloader()
model.eval()

true_list = []
corrupted_list = []
imputed_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

with torch.no_grad():
    for batch in tqdm(test_dl):
        x0, x1 = model.preprocess_batch(batch)
        x0 = x0.to(device)
        x1 = x1.to(device)
        node = NeuralODE(
            model.net, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        node = node.to(device)
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, 100, device=device),
            )

        tr = traj[-1].detach().cpu().numpy()
        x0 = x0.detach().cpu().numpy()
        x1 = x1.detach().cpu().numpy()
        true_list.append(x1)
        corrupted_list.append(x0)
        imputed_list.append(tr)

true = np.concatenate(true_list, axis=0)
corrupted = np.concatenate(corrupted_list, axis=0)
imputed = np.concatenate(imputed_list, axis=0)

#save the imputed data as csv files

mask = "data/split_data/test/klein/zero_four_dropout/masks/klein.npy"
mask = np.load(mask)

#apply the mask to the imputed data, true data and corrupted data
'''true = true[:,mask]
corrupted = corrupted[:,mask]
imputed = imputed[:,mask]'''

imputed[imputed<0.1] = 0

np.savetxt("imputed.csv", imputed, delimiter=",")
np.savetxt("corrupted.csv", corrupted, delimiter=",")
np.savetxt("true.csv", true, delimiter=",")

#print mse loss
print("MSE loss between imputed and true data")
print(np.mean((imputed - true)**2))

#print l1 loss
print("L1 loss between imputed and true data")
print(np.mean(np.abs(imputed - true)))

#get cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(imputed, true)
print("Cosine similarity between imputed and true data")
print(np.mean(cosine_similarity(imputed, true)))
