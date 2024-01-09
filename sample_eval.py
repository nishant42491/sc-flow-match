from tqdm import tqdm
import torch
from torchdyn.core import NeuralODE
from src.data.single_cell_datamodule import SingleCellDataModule
from src.models.cfm_module import CFMLitModule
from src.models.components.transformer_encoder import TransformerAutoencoder
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path
import numpy as np
import time





def compute_scores(imputed, true):

    rmse = np.sqrt(mean_squared_error(imputed, true))

    return rmse


def sample_dataset_task(dataset_name, task_name, num_timesteps=100):


    ckpt_path = fr"logs/train/runs/{dataset_name}/{task_name}/checkpoints/"

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
                            batch_size=2048,
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
                    t_span=torch.linspace(0, 1, num_timesteps, device=device),
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

    return imputed, true


def sample_time_vs_rmse(dataset_name, task_name):

    delta_t_list = []
    rmse_list = []
    labels = []

    for t in range(2, 100, 2):

        start_time = time.time()
        imputed, true = sample_dataset_task(dataset_name, task_name, t)
        end_time = time.time()
        delta_time = end_time - start_time

        err = imputed - true
        err = np.square(err)
        err = np.mean(err, axis=1)
        err = np.sqrt(err)
        rmse = np.mean(err)
        delta_t_list.append(delta_time)
        rmse_list.append(rmse)
        labels.append(t)


    return delta_t_list, rmse_list, labels




if __name__ == "__main__":


    dataset_name = ['klein']
    task_name = ['zero_four_dropout']

    for name in dataset_name:
        for task in task_name:
            delta_time, rmse, labels = sample_time_vs_rmse(name, task)


    import matplotlib.pyplot as plt
    plt.scatter(delta_time, rmse, c=labels, cmap='viridis')
    plt.title('Time vs RMSE')

    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.show()


