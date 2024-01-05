import numpy as np
import matplotlib.pyplot as plt


import os


os.chdir('../')
def extract_metric_values(baselines, datasets, metric, task):
    n = len(baselines) + 1
    m = len(datasets)

    data = np.zeros((n, m))  # Initialize a matrix to store metric values

    for i, baseline in enumerate(baselines):
        for j, dataset in enumerate(datasets):
            # Create the path based on the baseline, dataset, and task
            file_path = f"baseline_eval/{baseline}/{dataset}/{task}/{dataset}_cluster_results_log.txt"

            # Check if the file exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    # Read the file and extract the metric value
                    for line in file:
                        if line.startswith(f"{metric}:"):
                            metric_value = float(line.split(":")[1])
                            data[i, j] = metric_value
                            break  # Assuming there's only one metric value per file
            else:
                print(f"File not found: {file_path}")

    for j,ds in enumerate(datasets):

        file_path = f"eval_output/{ds}/{task}/{ds}_cluster_results_log.txt"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                # Read the file and extract the metric value
                for line in file:
                    if line.startswith(f"{metric}:"):
                        metric_value = float(line.split(":")[1])
                        data[-1, j] = metric_value
                        break



    return data



def create_heatmap(baselines, datasets, metric, task):
    n = len(baselines)
    m = len(datasets)

    # Generate random data for illustration purposes
    data = extract_metric_values(baselines,datasets,metric, task) # Replace this with your actual data
    data = np.array(data)
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', interpolation='nearest')

    baselines.append("CFM")
    n = len(baselines)

    # Adding labels and ticks
    plt.title(f"Heatmap for {metric}")
    plt.xlabel("Datasets")
    plt.ylabel("Baselines")
    plt.xticks(np.arange(m), datasets, rotation=75)
    plt.yticks(np.arange(n), baselines)

    # Adding values in cells
    for i in range(n):
        for j in range(m):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white')

    plt.colorbar(label='Values')
    plt.tight_layout()
    plt.show()


# Example usage:

if __name__ == "__main__":

    baselines = ["magic", "drimpute", "autoclass", "gnnimpute", "alra"]
    datasets = ["muraro",
                "plasschaert",
                "romanov",
                "tosches turtle",
                "young",
                "quake_10x_bladder",
                "quake_10x_limb_muscle",
                "quake_10x_spleen",
                "quake_smart-seq2_diaphragm",
                "quake_smart-seq2_heart",
                "quake_smart-seq2_limb_muscle",
                "quake_smart-seq2_lung",
                "quake_smart-seq2_trachea"]
    metric = 'ARI'
    task = 'zero_four_dropout'

    create_heatmap(baselines, datasets, metric, task)
