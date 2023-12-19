from torch.utils.data import DataLoader,TensorDataset
from lightning import LightningDataModule
import torch
import os
import pandas as pd
#import scikitlearn minmax scaler
from sklearn.preprocessing import MinMaxScaler


class SingleCellDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data/",
                 batch_size: int = 64,
                 name: str = "klein",
                 task: str = "zero_four_dropout",
                 num_samples = None) -> None:



        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.name = name
        self.task = task
        self.train_dataset, self.test_dataset = self.load_train_test_data(num_samples)
        self.train_min_max_scaler = MinMaxScaler()
        self.test_min_max_scaler = MinMaxScaler()
        self.dim = self.train_dataset[0][0].shape[0]


    def load_train_test_data(self, num_samples = None):

        train_data_original_path = os.path.join(self.data_dir, "split_data", "train",
                                                f"{self.name}", "original", f'{self.name}.csv'
                                                )

        train_data_corrupted_path = os.path.join(self.data_dir, "split_data", "train",
                                                 f"{self.name}", f"{self.task}", f'{self.name}.csv'
                                                 )

        test_data_original_path = os.path.join(self.data_dir, "split_data", "test",
                                               f"{self.name}", "original", f'{self.name}.csv'
                                               )

        test_data_corrupted_path = os.path.join(self.data_dir, "split_data", "test",
                                                f"{self.name}", f"{self.task}", f'{self.name}.csv'
                                                )


        if num_samples == None:

            train_data_og = pd.read_csv(train_data_original_path, header=None).values
            train_data_corrupted = pd.read_csv(train_data_corrupted_path, header=None).values

        else:
            train_data_og = pd.read_csv(train_data_original_path, header=None).values[:num_samples]
            train_data_corrupted = pd.read_csv(train_data_corrupted_path, header=None).values[:num_samples]

        test_data_og = pd.read_csv(test_data_original_path, header=None).values
        test_data_corrupted = pd.read_csv(test_data_corrupted_path, header=None).values
        train_data = TensorDataset(torch.tensor(train_data_corrupted, dtype=torch.float32),
                                   torch.tensor(train_data_og, dtype=torch.float32))
        test_data = TensorDataset(torch.tensor(test_data_corrupted, dtype=torch.float32),
                                    torch.tensor(test_data_og, dtype=torch.float32))
        return train_data, test_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)




if __name__ == "__main__":
    #write driver code for the datamodule
    data_dir = r"D:\Nishant\cfm_impute\flow-matching-single cell\data"
    batch_size = 64
    name = "klein"
    task = "zero_four_dropout"
    dm = SingleCellDataModule(data_dir, batch_size, name, task)
    print(dm.dim)
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()




