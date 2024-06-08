The code here implements Conditional Flow Matching for Single-Cell RNA Sequencing Impputations. We follow the Independent Confitional Flow Matching Methodology proposed by Tong et al.

TO RUN THE CODE:

1> Set up an enviornment using virtualenv and install the required packages look at https://github.com/ashleve/lightning-hydra-template install those requirements then install more requirements as required.
<br>
2> Place your data in the root data folder with the structure as data/split_data/{train/test}/{dataset_name}.<br>2.1> data/split_data/{train/test}/{dataset_name} should have further directories like: 
<br>
<br>
data/split_data/train/{dataset_name}/original<br>
data/split_data/test/{dataset_name}/zero_one_dropout<br>
data/split_data/test/{dataset_name}/zero_two_dropout<br> 
data/split_data/test/{dataset_name}/zero_four_dropout<br>

In the data/split_data/train/{dataset_name}/original it should contain a {dataset_name}.csv and a labels directory with {dataset_name}_labels.csv file in it.

In the data/split_data/train/{dataset_name}/zero_{percantage}_dropout folder should contain a {dataset_name}.csv and a masks directory with {dataset_name}.npy mask file which ahs been applied to the data to simulate dropout effect.

A sample structure is present in the data folder.

<br>

3> Run the scrits/new_schedule.sh with the required task name and dataset name. Trained weights should be in the logs folder.

4> Run the sample.py file by making the neccesary changes in the sample.py file to load the model and run the imputation on the test data.

5> The results will be saved in the eval_output folder for the clustering and imputation results.

Plotting functionalities are available in the plot_code folder.







