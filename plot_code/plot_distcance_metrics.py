import numpy as np
import matplotlib.pyplot as plt


import os


os.chdir('../')
def extract_metric_values(baselines, dataset, metric, tasks):
    n = len(baselines) + 1
    m = len(tasks)

    data = np.zeros((n, m))  # Initialize a matrix to store metric values

    for i, baseline in enumerate(baselines):
        for j, task in enumerate(tasks):
            # Create the path based on the baseline, dataset, and task
            file_path = f"baseline_eval/{baseline}/{dataset}/{task}/{dataset}_results_log.txt"

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

    for j,task in enumerate(tasks):

        file_path = f"eval_output/{dataset}/{task}/{dataset}_results_log.txt"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                # Read the file and extract the metric value
                for line in file:
                    if line.startswith(f"{metric}:"):
                        metric_value = float(line.split(":")[1])
                        data[-1, j] = metric_value
                        break

    return data

def create_heatmap(baselines, dataset, metric, tasks):
    n = len(baselines)
    m = len(tasks)

    # Generate random data for illustration purposes
    data = extract_metric_values(baselines,dataset,metric, tasks) # Replace this with your actual data
    data = np.array(data)
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    #change the colourmap to blue green thingie

    plt.imshow(data, cmap='viridis', interpolation='nearest')
    # plt.imshow(data, cmap='Blues', interpolation='nearest')

    baselines.append("CFM")
    n = len(baselines)

    # Adding labels and ticks
    plt.title(f"Heatmap for {metric} ({dataset})")
    plt.xlabel("Dataset-Task")
    plt.ylabel("Baselines")
    plt.xticks(np.arange(m), tasks, rotation=90)
    plt.yticks(np.arange(n), baselines)

    # Adding values in cells
    for i in range(n):
        for j in range(m):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black')

    plt.colorbar(label='Values')
    plt.tight_layout()
    #plt.show()

    #save plot in the ./plots/distances folder with name of dataset and metric
    plt.savefig(f'./plots/distance/{dataset}_{metric}.png')



if __name__ == "__main__":

    if __name__ == "__main__":
        baselines = ["magic", "drimpute", "autoclass"]
        dataset = "ziesel"
        metric = 'RMSE'
        tasks = ['zero_two_dropout', 'zero_one_dropout', 'zero_four_dropout']

        for dataset in ['ziesel', 'klein']:
            for metric in ['RMSE', 'Cosine_Similarity', 'L1_Distance']:
                create_heatmap(baselines.copy(), dataset, metric, tasks)
