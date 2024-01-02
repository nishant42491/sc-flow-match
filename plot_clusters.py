import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_name = 'muraro'
original_train = pd.read_csv(f'./data/split_data/train/{dataset_name}/original/{dataset_name}.csv', header=None)
original_test = pd.read_csv(f'./data/split_data/test/{dataset_name}/original/{dataset_name}.csv', header=None)
train_labels = pd.read_csv(f'./data/split_data/train/{dataset_name}/original/labels/{dataset_name}_labels.csv', header=None)
test_labels = pd.read_csv(f'./data/split_data/test/{dataset_name}/original/labels/{dataset_name}_labels.csv', header=None)

#merge the train and test data
original_df = pd.concat([original_train, original_test])
labels_df = pd.concat([train_labels, test_labels])

#plot 2 dimensional data UMAP plots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns

data = original_df.values
labels = labels_df.values

#plot TSNE plots

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(data)
tsne_df = pd.DataFrame(tsne_results)
tsne_df['labels'] = labels
tsne_df.columns = ['x','y','labels']
tsne_df['labels'] = tsne_df['labels'].astype(str)
num_labels = len(np.unique(labels))
plt.figure()
sns.scatterplot(
    x="x", y="y",
    hue="labels",
    palette=sns.color_palette("hls", num_labels),
    data=tsne_df,
    legend="full",
    alpha=0.3
)
from pathlib import Path
Path(f'./plots/tsne_plots').mkdir(parents=True, exist_ok=True)
plt.savefig(f'./plots/tsne_plots/{dataset_name}_tsne.png')