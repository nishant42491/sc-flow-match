import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import torch
from torchdyn.core import NeuralODE
from src.data.single_cell_datamodule import SingleCellDataModule

from src.models.cfm_module import CFMLitModule
from src.models.components.transformer_encoder import TransformerAutoencoder
from pathlib import Path


def sample_dataset_task(dataset_name,
                        task_name,
                        ckpt_dataset_name,
                        ckpt_task_name):


    ckpt_path = fr"logs/train/runs/{ckpt_dataset_name}/{ckpt_task_name}/checkpoints/"

    for fname in os.listdir(ckpt_path):
        if "last" not in fname:
            ckpt_path = os.path.join(ckpt_path, fname)
            break
    #print the lastname of the ckpt file
    print(ckpt_path)

    model = CFMLitModule.load_from_checkpoint(ckpt_path,
                                            net=TransformerAutoencoder(input_dim=2048,
                                                                       embed_dim=128,
                                                                       time_dim=32,
                                                                       num_layers=1,
                                                                       n_heads=1,
                                                                       output_dim=2048,
                                                                       dropout=0.1))

    dm = SingleCellDataModule(data_dir=r"D:\Nishant\cfm_impute\flow-matching-single cell\data",
                            batch_size=4096,
                            name=dataset_name,
                            task=task_name)

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

    imputed[imputed < 0.1] = 0

    #calculate rmse
    cs_l = []
    for i in range(len(imputed)):
        cs = cos_sim(true[i], imputed[i])
        cs_l.append(cs)

    #averege cosine similarity
    cs = np.mean(np.array(cs_l))

    return cs

def cos_sim(a, b):

    z = np.dot(a, b)
    x = np.linalg.norm(a)
    y = np.linalg.norm(b)
    return z/(x*y)

if __name__ == "__main__":

    os.chdir("../")

    ckpt_dataset_name = ["klein","ziesel",
                         'muraro', 'quake_10x_bladder', 'romanov',
                         ]
    ckpt_task_name = "zero_two_dropout"
    task_name = 'zero_two_dropout'
    dataset_name = [ "young",
                     "quake_smart-seq2_lung",
                     "tosches turtle",
                     "quake_10x_spleen",]

    r_list = []
    for ckpt_dnmae in ckpt_dataset_name:
        c_list = []
        for name in dataset_name:

            rmse = sample_dataset_task(name,
                                task_name,
                                ckpt_dnmae,
                                ckpt_task_name
                                )

            print(rmse)

            c_list.append(rmse)

        r_list.append(c_list)


    #plot the lists in r_list as lineplots with x axis as dataset_name and y axis as rmse

    for i in range(len(r_list)):
        plt.plot(dataset_name,r_list[i],label=ckpt_dataset_name[i])

    plt.xlabel("Dataset")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.show()





