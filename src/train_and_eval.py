import os
import numpy as np
import torch
from missing_patterns import *
from pygrinder import mcar
from load_local_dataset import get_dataset, available_datasets
from sklearn.model_selection import train_test_split
from pypots.imputation import DLinear, PatchTST, SAITS
from Guided.imputation import CSDI
from pypots.utils.metrics import calc_mae
from imputation.csdi_patchtst.model import CSDI_PatchTST
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
parser.add_argument('--diffusion_steps', type = int, default = 30)
parser.add_argument('--train_all', type = bool, default = False)
parser.add_argument('--cls_free', type = bool, default = False)
parser.add_argument('--drop_prob_tot', type = float, default = None)
parser.add_argument('--drop_prob_iid', type = float, default = None)
parser.add_argument('--guidance_str', type = float, default = None)
parser.add_argument('--n_sampling_times', type = int, default = 1)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--schedule', type = str, default = 'linear')
args = parser.parse_args()
print("ARGS: ", args)

dir_path = os.path.join(f"../logs_new/{get_norm(args)}")
if args.cls_free:
    dir_path = os.path.join(dir_path, "classifier_free")
    dir_path = os.path.join(dir_path, args.schedule)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
logpath = os.path.join(dir_path, f"diff_{args.diffusion_steps}.txt")
logfile = open(logpath, 'w')

#dataset_names = ['chlorine', 'electricity']
dataset_names = ['chlorine']
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

        if args.train_all:

            saits = SAITS(
                n_steps=50, n_features=num_features, n_layers=4,
                d_model=256, d_ffn=128, n_heads=4, d_k=64, d_v=64,
                dropout=0.1, epochs=args.epochs, saving_path=f"../{function_name}/{dataset_name}/saits",
                model_saving_strategy="best"
            )

            patchtst = PatchTST(
                n_steps=50, n_features=num_features, patch_len=5, stride=5,
                n_layers=4, n_heads=4, d_k=64, d_v=64, d_model=256, d_ffn=128,
                dropout=0.1, attn_dropout=0.05, saving_path=f"../{function_name}/{dataset_name}/patchtst",
                model_saving_strategy="best", epochs=args.epochs
            )

            dlinear = DLinear(
                n_steps=50, n_features=num_features, moving_avg_window_size=3,
                d_model=256, epochs=args.epochs, saving_path=f"../{function_name}/{dataset_name}/dlinear",
                model_saving_strategy="best"
            )

        print("initializing csdi")

        if not args.cls_free:
            csdi = CSDI(
                n_steps = 50, n_features=num_features, n_layers=4, n_heads=4, n_channels=8,
                d_time_embedding=32, d_feature_embedding=32, d_diffusion_embedding=128,
                n_diffusion_steps = args.diffusion_steps, schedule = args.schedule,
                epochs=args.epochs, saving_path=f"../{function_name}/{dataset_name}/csdi",
                model_saving_strategy="best", is_cls_free_guided= args.cls_free, 
                cond_drop_prob_total = args.drop_prob_tot, cond_drop_prob_individual= args.drop_prob_iid,
                guidance_strength= args.guidance_str
            )
            model_list = [(saits, 'SAITS'), (patchtst, 'PatchTST'), (dlinear, 'DLinear'), (csdi, 'CSDI')]
            if not args.train_all:
                model_list = [(csdi, 'CSDI')]
        
        else:
            guidance_strengths = [0.1, 0.5, 1.0, 2.0] if args.guidance_str is None else [args.guidance_str]
            drop_prob_tot = [0.05 + i*0.05 for i in range(5)] if args.drop_prob_tot is None else [args.drop_prob_tot]
            drop_prob_iid = [0.05 + i*0.05 for i in range(5)] if args.drop_prob_iid is None else [args.drop_prob_iid]
            model_list = []
            for gd_str in guidance_strengths:
                for drp_tot in drop_prob_tot:
                    for drp_iid in drop_prob_iid:
                        csdi_model = CSDI(
                            n_steps = 50, n_features=num_features, n_layers=4, n_heads=4, n_channels=8,
                            d_time_embedding=32, d_feature_embedding=32, d_diffusion_embedding=128,
                            n_diffusion_steps = args.diffusion_steps, schedule= args.schedule,
                            epochs=args.epochs, saving_path=f"../{function_name}/{dataset_name}/csdi",
                            model_saving_strategy="best", is_cls_free_guided= args.cls_free, 
                            cond_drop_prob_total = drp_tot, cond_drop_prob_individual= drp_iid,
                            guidance_strength= gd_str
                        )
                        model_list.append((csdi_model, f"CSDI guidance {gd_str} drop prob total {drp_tot} drop prob iid {drp_iid}"))

            vanilla_pair = (CSDI(
                n_steps = 50, n_features=num_features, n_layers=4, n_heads=4, n_channels=8,
                d_time_embedding=32, d_feature_embedding=32, d_diffusion_embedding=128,
                n_diffusion_steps = args.diffusion_steps, schedule = args.schedule,
                epochs=args.epochs, saving_path=f"../{function_name}/{dataset_name}/csdi",
                model_saving_strategy="best", is_cls_free_guided= False
            ), "CSDI Vanilla")
            model_list.append(vanilla_pair)

        for model, model_name in model_list:
            print(f"Training MODEL: {model_name} on dataset: {dataset_name} with BLOCK FN: {function_name}", file = logfile)
            if model_name[:4] == 'CSDI':
                print('csdi')
                model.fit(train_ds, n_sampling_times= args.n_sampling_times)
            else:
                model.fit(train_ds)
            print("Training concluded", file = logfile)
            imputation = model.impute(test_ds)
            indicating_mask = reshaped_dataset.isnan()
            mae = calc_mae(torch.tensor(imputation).squeeze(), unmasked_reshaped_dataset, indicating_mask)
            print(f"Test MAE: {mae}", file = logfile)

        print('\n', file = logfile)

logfile.close()
