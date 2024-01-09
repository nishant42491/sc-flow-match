import pandas as pd
import numpy as np
import os
#import tsne
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


if __name__ == '__main__':

    os.chdir('../')

    dataset_name = "romanov"
    task_name = "zero_four_dropout"

    original_data_train = pd.read_csv(f"data/split_data/train/{dataset_name}/original/{dataset_name}.csv", header=None).values
    train_labels = pd.read_csv(f"data/split_data/train/{dataset_name}/original/labels/{dataset_name}_labels.csv",header=None).values

    generated_data_og = pd.read_csv(f"outputs/{dataset_name}/{task_name}/og_out.csv", header=None).values
    test_labels = pd.read_csv(f"data/split_data/test/{dataset_name}/original/labels/{dataset_name}_labels.csv", header=None).values
    generated_data = pd.read_csv(f"outputs/{dataset_name}/{task_name}/gen_out.csv", header=None).values

    baseline_magic = pd.read_csv(f"baseline_outputs/magic/{dataset_name}/{task_name}/{dataset_name}_magic_out.csv", header=None).values
    baeline_alra = pd.read_csv(f"baseline_outputs/alra/{dataset_name}/{task_name}/{dataset_name}_imputed.csv",skiprows=1, header=None).values
    baseline_gnnimpute = pd.read_csv(f"baseline_outputs/gnnimpute/{dataset_name}/{task_name}/gnnimpute_out.csv", header=None).values
    baseline_drimpute = pd.read_csv(f"baseline_outputs/drimpute/{dataset_name}/{task_name}/{dataset_name}_imputed.csv").values
    #transpose the autoclass output
    baseline_drimpute = baseline_drimpute.T
    baseline_autoclass = pd.read_csv(f"baseline_outputs/autoclass/{dataset_name}/{task_name}/autoclass_imputed.csv", header=None).values




    #create TSNE plot for test_labels
    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(generated_data_og)
    tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'label':test_labels[:,0]})

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x="X", y="Y",
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=tsne_df,
        legend="full",
        alpha=0.3
    )

    #plt.savefig at plots/clustering/tsne_original.png
    plt.savefig(f'plots/clustering/tsne_original.png')

    #create TSNE plot for generated data
    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(generated_data)
    tsne_df_gen = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'label':test_labels[:,0]})



    #plt is using show
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="X", y="Y",
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=tsne_df_gen,
        legend="full",
        alpha=0.3
    )

    #plt.savefig at plots/clustering/tsne_generated.png
    plt.savefig(f'plots/clustering/tsne_generated.png')

    #create tsne plots for baselines
    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(baseline_magic)
    tsne_df_magic = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'label':test_labels[:,0]})
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="X", y="Y",
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=tsne_df_magic,
        legend="full",
        alpha=0.3
    )

    plt.savefig(f'plots/clustering/tsne_baseline_magic.png')

    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(baeline_alra)
    tsne_df_alra = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'label':test_labels[:,0]})
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="X", y="Y",
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=tsne_df_alra,
        legend="full",
        alpha=0.3
    )
    plt.savefig(f'plots/clustering/tsne_baseline_alra.png')

    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(baseline_gnnimpute)
    tsne_df_gnnimpute = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'label':test_labels[:,0]})
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="X", y="Y",
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=tsne_df_gnnimpute,
        legend="full",
        alpha=0.3
    )
    plt.savefig(f'plots/clustering/tsne_baseline_gnnimpute.png')

    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(baseline_drimpute)
    tsne_df_drimpute = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'label':test_labels[:,0]})
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="X", y="Y",
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=tsne_df_drimpute,
        legend="full",
        alpha=0.3
    )
    plt.savefig(f'plots/clustering/tsne_baseline_drimpute.png')

    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(baseline_autoclass)
    tsne_df_autoclass = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'label':test_labels[:,0]})
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="X", y="Y",
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=tsne_df_autoclass,
        legend="full",
        alpha=0.3
    )
    plt.savefig(f'plots/clustering/tsne_baseline_autoclass.png')







