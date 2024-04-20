import os
import numpy as np
import torch
DATASET_DIR = '../datasets/'
available_datasets = os.listdir(DATASET_DIR)

def get_dataset(name):
    assert name in available_datasets, f"unknown dataset {name}"
    return torch.tensor(np.loadtxt(os.path.join(DATASET_DIR, name, f'{name}_normal.txt')))

if __name__ == "__main__":
    for dataset in available_datasets:
        data_arr = get_dataset(dataset)
        print(dataset, f'Shape: {data_arr.shape}', f'Availability: {(data_arr == data_arr).float().mean()}', sep=' | ')
