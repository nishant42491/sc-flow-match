import math
import os

from tqdm import tqdm
import numpy as np
import torch
from torchdyn.core import NeuralODE
from src.data.single_cell_datamodule import SingleCellDataModule

from src.models.cfm_module import CFMLitModule
from src.models.components.transformer_encoder import TransformerAutoencoder
from pathlib import Path


def sample_dataset_task(dataset_name, task_name, output_dir):


    ckpt_path = fr"logs/train/runs/{dataset_name}/{task_name}/checkpoints/"
    #in the ckpt_path there are two .ckpt files ones named last.ckpt. choos the file not named last.ckpt
    #ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[1])

    for fname in os.listdir(ckpt_path):
        if "last" not in fname:
            ckpt_path = os.path.join(ckpt_path, fname)
            break
    #print the lastname of the ckpt file
    print(ckpt_path)

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

    true_save_dir = f"{output_dir}/{dataset_name}/{task_name}/og_out.csv"
    corrupted_save_dir = f"{output_dir}/{dataset_name}/{task_name}/corrupted.csv"
    imputed_save_dir = f"{output_dir}/{dataset_name}/{task_name}/gen_out.csv"

    Path(true_save_dir).parent.mkdir(parents=True, exist_ok=True)
    Path(corrupted_save_dir).parent.mkdir(parents=True, exist_ok=True)
    Path(imputed_save_dir).parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(true_save_dir, true, delimiter=",")
    np.savetxt(corrupted_save_dir, corrupted, delimiter=",")
    np.savetxt(imputed_save_dir, imputed, delimiter=",")

if __name__ == "__main__":

    dataset_name = ['muraro','plasschaert','romanov','tosches turtle',
                    "young", "quake_10x_bladder","quake_10x_limb_muscle", "quake_10x_spleen",
                    "quake_smart-seq2_diaphragm", "quake_smart-seq2_heart", "quake_smart-seq2_limb_muscle",
                    "quake_smart-seq2_lung", "quake_smart-seq2_trachea"]
    task_name = ['zero_one_dropout','zero_two_dropout','zero_four_dropout']
    output_dir = './outputs'
    for name in dataset_name:
        for task in task_name:
            sample_dataset_task(name,task,output_dir)

