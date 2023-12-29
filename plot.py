import os
import matplotlib.pyplot as plt
import numpy as np

def get_data_from_dirs(dataset_name, task_name, method_dir="eval_output", baseline_dir="baseline_eval"):
    # Function to extract data from the text files
    def extract_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = {}
            for line in lines:
                line = line.strip().split(': ')
                if len(line) == 2:
                    if line[0] == "Dataset":
                        data[line[0]] = line[1]
                    else:
                        data[line[0]] = float(line[1])
            return data

    # Function to fetch data from directories
    def fetch_data_from_directory(directory):
        data = {}
        file_path = os.path.join(directory, f"{dataset_name}_results_log.txt")
        root = os.path.normpath(directory)
        parts = root.split(os.path.sep)
        baseline_name = parts[-3]  # Extracting baseline name from path
        if parts[-2] == dataset_name and parts[-1] == task_name:
            data[baseline_name] = extract_data(file_path)
        return data

    # Fetch data for your method
    your_method_dir = os.path.join(method_dir, dataset_name, task_name)
    your_method_data = fetch_data_from_directory(your_method_dir)

    baseline_names = os.listdir(baseline_dir)

    for baseline_name in baseline_names:
        baseline_data_dir = os.path.join(baseline_dir, baseline_name, dataset_name, task_name)
        baseline_data = fetch_data_from_directory(baseline_data_dir)
        your_method_data = your_method_data | baseline_data

        return your_method_data

def plot_heatmap_with_values(data_dict):

    methods = list(data_dict.keys())
    metrics = list(next(iter(data_dict.values())).keys())
    #drop the dataset name across all methods
    data_dict = {method: {metric: data_dict[method][metric] for metric in metrics[1:]} for method in methods}
    metrics = metrics[1:]


    data = np.array([[data_dict[method][metric] for metric in metrics] for method in methods])
    #data = data[:, 1:]
    data = np.float32(data)
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')

    for i in range(len(methods)):
        for j in range(len(metrics)):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white')

    plt.colorbar(label='Metric Values')
    plt.xticks(np.arange(len(metrics)), metrics)
    plt.yticks(np.arange(len(methods)), methods)
    plt.xlabel('Metrics')
    plt.ylabel('Methods')
    plt.title('Heatmap of Experiment Metrics Across Methods')
    plt.tight_layout()
    plt.show()

d = get_data_from_dirs("ziesel", "zero_four_dropout")
plot_heatmap_with_values(d)


