import os
import numpy as np
import torch
import torch.nn.functional as F
DATASET_DIR = '../datasets'
available_datasets = os.listdir(DATASET_DIR)
print(available_datasets)

def get_dataset(name, norm = None):
    #print(f"Normalization type: {norm}")
    assert name in available_datasets, f"unknown dataset {name}"
    print(os.getcwd())
    raw_data = torch.tensor(np.loadtxt(os.path.join(DATASET_DIR, name, f'{name}_normal.txt')))
    if norm is None:
        return raw_data
    elif norm == 'max':
        dataset = F.normalize(raw_data, dim = 0)
        return dataset
    else:
        means = torch.mean(raw_data, dim = 0)
        assert means.shape[0] == raw_data.shape[1]
        devs = torch.std(raw_data, dim = 0)
        assert means.shape == devs.shape
        dataset = torch.div(raw_data - means, devs + 1e-12)
        return dataset
        


if __name__ == "__main__":
    for dataset in available_datasets:
        data_arr_none = get_dataset(dataset, None)
        data_arr_max = get_dataset(dataset, 'max')
        data_arr_zero = get_dataset(dataset, 'zero')
        #print("Type: ", type(data_arr))
        print(dataset, f'Shape: {data_arr_none.shape}', f'Availability: {(data_arr_none == data_arr_none).float().mean()}', sep=' | ')
        print(f'No norm means: {torch.mean(data_arr_none, 0)}')
        print(f'Max norm means: {torch.mean(data_arr_max, 0)}')
        print(f'Zero norm means: {torch.mean(data_arr_zero, 0)}')