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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score



def evaluate_clusters(original_csv, resultant_csv,dataset_name, dropout_eval_name):

    original_df = pd.read_csv(original_csv, header=None)
    resultant_df = pd.read_csv(resultant_csv, header=None)

    test_labels = pd.read_csv(f'./data/split_data/test/{dataset_name}/original/labels/{dataset_name}_labels.csv', header=None)

    num_unique_labels = len(np.unique(test_labels.values))

    kmeans = KMeans(n_clusters=num_unique_labels, random_state=42).fit(original_df.values)
    original_labels = kmeans.labels_

    resultant_labels = kmeans.predict(resultant_df.values)

    ari = adjusted_rand_score(original_labels, resultant_labels)
    nmi = normalized_mutual_info_score(original_labels, resultant_labels)






def sample_time_vs_rmse(dataset_name, task_name):

    delta_t_list = []
    rmse_list = []

    ckpt_path = fr"logs/train/runs/{dataset_name}/{task_name}/checkpoints/"

    for fname in os.listdir(ckpt_path):
        if "last" not in fname:
            ckpt_path = os.path.join(ckpt_path, fname)
            break
    # print the lastname of the ckpt file
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





    for t in [20,20, 50, 100, 200, 500, 1000]:

        start_time = time.time()

        #####################################

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
                        t_span=torch.linspace(0, 1, t, device=device),
                    )

                tr = traj[-1].detach().cpu().numpy()
                x0 = x0.detach().cpu().numpy()
                x1 = x1.detach().cpu().numpy()
                true_list.append(x1)
                corrupted_list.append(x0)
                imputed_list.append(tr)
                true = np.concatenate(true_list, axis=0)
                imputed = np.concatenate(imputed_list, axis=0)

                imputed[imputed < 0.1] = 0


        #####################################

        end_time = time.time()
        delta_time = end_time - start_time

        '''err = imputed - true
        err = np.square(err)
        err = np.mean(err, axis=1)
        err = np.sqrt(err)
        rmse = np.mean(err)'''
        #calculate cosine similarity

        css = compute_scores(imputed, true)


        delta_t_list.append(delta_time)
        rmse_list.append(css)


    return delta_t_list, rmse_list




if __name__ == "__main__":


    dataset_name = ['ziesel', 'young', 'tosches turtle']
    task_name = ['zero_two_dropout']

    marker_names = ["20", "50", "100", "200", "500"]

    delta_times= []
    rmses = []

    for dataset in dataset_name:

        delta_time, rmse = sample_time_vs_rmse(dataset, task_name[0])
        rmse = np.array(rmse)
        rmse = (rmse - rmse.min()) / (rmse.max() - rmse.min())
        delta_time = delta_time[1:]
        rmse = rmse[1:]
        delta_times.append(delta_time)
        rmses.append(rmse)

    #delta_time, rmse = sample_time_vs_rmse(dataset_name[0], task_name[0])

    import matplotlib.pyplot as plt

    for i in range(len(delta_times)):

        #plot line plots and use diffente colours for different dataasets
        plt.plot(delta_times[i], rmses[i], label=dataset_name[i])

        #plot legend

        plt.legend()


        for j, txt in enumerate(marker_names):
            plt.annotate(txt, (delta_times[i][j], rmses[i][j]))

        plt.xlabel('Time')
        plt.ylabel('RMSE')
    plt.show()



