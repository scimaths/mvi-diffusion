import numpy as np
import torch
from missing_patterns import *
from pygrinder import mcar
from load_local_dataset import get_dataset, available_datasets
from sklearn.model_selection import train_test_split
from pypots.imputation import CSDI, DLinear, PatchTST, SAITS
from pypots.utils.metrics import calc_mae

for dataset_name in ['electricity', 'chlorine', 'temp', 'meteo']:
    dataset = get_dataset(dataset_name)
    for blocking_function, function_name in [
        (blackout, 'blackout'),
        (block_mcar, 'block_mcar'),
    ]:
        masked_dataset = blocking_function(dataset)
        print(dataset_name, masked_dataset.isnan().float().mean(), masked_dataset.shape)
        series_len, num_features = masked_dataset.shape
        unmasked_reshaped_dataset = dataset.reshape(-1, 50, num_features)
        reshaped_dataset = masked_dataset.reshape(-1, 50, num_features)
        train_ds, val_ds = train_test_split(reshaped_dataset, test_size=0.1)
        print(train_ds.shape, val_ds.shape)
        train_ds = {'X': reshaped_dataset}
        val_ds = {'X': val_ds}
        test_ds = {'X': reshaped_dataset}

        saits = SAITS(
            n_steps=50, n_features=num_features, n_layers=4,
            d_model=256, d_ffn=128, n_heads=4, d_k=64, d_v=64,
            dropout=0.1, epochs=100, saving_path=f"../{function_name}/{dataset_name}/saits",
            model_saving_strategy="best"
        )

        patchtst = PatchTST(
            n_steps=50, n_features=num_features, patch_len=5, stride=5,
            n_layers=4, n_heads=4, d_k=64, d_v=64, d_model=256, d_ffn=128,
            dropout=0.1, attn_dropout=0.05, saving_path=f"../{function_name}/{dataset_name}/patchtst",
            model_saving_strategy="best", epochs=100
        )

        dlinear = DLinear(
            n_steps=50, n_features=num_features, moving_avg_window_size=3,
            d_model=256, epochs=100, saving_path=f"../{function_name}/{dataset_name}/dlinear",
            model_saving_strategy="best"
        )

        csdi = CSDI(
            n_features=num_features, n_layers=4, n_heads=4, n_channels=8,
            d_time_embedding=32, d_feature_embedding=32, d_diffusion_embedding=128,
            epochs=100, saving_path=f"../{function_name}/{dataset_name}/csdi",
            model_saving_strategy="best"
        )

        for model, model_name in [
            (saits, 'SAITS'),
            (patchtst, 'PatchTST'),
            (dlinear, 'DLinear'),
            (csdi, 'CSDI')
        ]:
            print(f"Training model {model_name} on {dataset_name}; {function_name}")
            model.fit(train_ds)
            print("Training concluded")
            imputation = model.impute(test_ds)
            indicating_mask = reshaped_dataset.isnan()
            mae = calc_mae(torch.tensor(imputation).squeeze(), unmasked_reshaped_dataset, indicating_mask)
            print(f"Test MAE - {mae}")
