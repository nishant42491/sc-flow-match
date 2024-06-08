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







