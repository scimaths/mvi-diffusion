import numpy as np
import torch
from missing_patterns import *
from pygrinder import mcar
from load_local_dataset import get_dataset, available_datasets
from sklearn.model_selection import train_test_split
from pypots.imputation import CSDI, DLinear, PatchTST, SAITS
from pypots.utils.metrics import calc_mae
from argparse import ArgumentParser
import os
from tqdm import tqdm

def get_norm(args):
    if args.normtype is None:
        return 'un_norm'
    else:
        return 'max_norm'

parser = ArgumentParser()
parser.add_argument('--normtype', type = str, default = None)
parser.add_argument('--diffusion_steps', type = int, default = 50)
parser.add_argument('--train_all', type = bool, default = False)
args = parser.parse_args()
print("ARGS: ", args)
dir_path = os.path.join(f"../logs_new/{get_norm(args)}")
logpath = os.path.join(dir_path, f"diff_{args.diffusion_steps}.txt")
logfile = open(logpath, 'w')

dataset_names = ['chlorine', 'electricity', 'temp', 'meteo', 'airq']
#dataset_names = ['chlorine']
for dataset_name in tqdm(dataset_names):
    dataset = get_dataset(dataset_name, args.normtype)
    for blocking_function, function_name in [
        (blackout, 'blackout'),
        (block_mcar, 'block_mcar'),
    ]:
        
        masked_dataset = blocking_function(dataset)
        print(f"DATASET: {dataset_name}", file = logfile)
        print(f"Missing percentage: {masked_dataset.isnan().float().mean()} Masked data shape: {masked_dataset.shape}", file = logfile)
        series_len, num_features = masked_dataset.shape
        unmasked_reshaped_dataset = dataset.reshape(-1, 50, num_features)
        reshaped_dataset = masked_dataset.reshape(-1, 50, num_features)
        train_ds, val_ds = train_test_split(reshaped_dataset, test_size=0.1)
        print(f"Train data shape: {train_ds.shape} Val data shape: {val_ds.shape}", file = logfile)
        train_ds = {'X': reshaped_dataset}
        val_ds = {'X': val_ds}
        test_ds = {'X': reshaped_dataset}

        saits = SAITS(
            n_steps=50, n_features=num_features, n_layers=4,
            d_model=256, d_ffn=128, n_heads=4, d_k=64, d_v=64,
            dropout=0.1, epochs=300, saving_path=f"../{function_name}/{dataset_name}/saits",
            model_saving_strategy="best"
        )

        patchtst = PatchTST(
            n_steps=50, n_features=num_features, patch_len=5, stride=5,
            n_layers=4, n_heads=4, d_k=64, d_v=64, d_model=256, d_ffn=128,
            dropout=0.1, attn_dropout=0.05, saving_path=f"../{function_name}/{dataset_name}/patchtst",
            model_saving_strategy="best", epochs=300
        )

        dlinear = DLinear(
            n_steps=50, n_features=num_features, moving_avg_window_size=3,
            d_model=256, epochs=300, saving_path=f"../{function_name}/{dataset_name}/dlinear",
            model_saving_strategy="best"
        )

        csdi = CSDI(
            n_features=num_features, n_layers=4, n_heads=4, n_channels=8,
            d_time_embedding=32, d_feature_embedding=32, d_diffusion_embedding=128,
            n_diffusion_steps = args.diffusion_steps,
            epochs=300, saving_path=f"../{function_name}/{dataset_name}/csdi",
            model_saving_strategy="best"
        )
        model_list = [(saits, 'SAITS'), (patchtst, 'PatchTST'), (dlinear, 'DLinear'), (csdi, 'CSDI')]
        if not args.train_all:
            model_list = [(csdi, 'CSDI')]

        for model, model_name in model_list:
            print(f"Training MODEL: {model_name} on dataset: {dataset_name} with BLOCK FN: {function_name}", file = logfile)
            model.fit(train_ds)
            print("Training concluded", file = logfile)
            imputation = model.impute(test_ds)
            indicating_mask = reshaped_dataset.isnan()
            mae = calc_mae(torch.tensor(imputation).squeeze(), unmasked_reshaped_dataset, indicating_mask)
            print(f"Test MAE: {mae}", file = logfile)

        print('\n', file = logfile)

logfile.close()
