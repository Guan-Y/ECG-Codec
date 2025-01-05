from cross_patients_performance import Get_All_Result
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
import os.path as osp
from glob import glob
from tqdm import tqdm
import os
from models.model import EncodecModel
import torch


rst_files = glob("./experiments/Encodec-Seanet/Thu-Jul-25-13_51_42-2024/*/result/*")
rst_files_projected = glob("./experiments/Encodec-Seanet-lookup/Thu-Jul-25-13_51_42-2024/*/result/*")

src_ckpt_dir = "./experiments/Encodec-Seanet/Thu-Jul-25-13_51_42-2024"
src_ckpt_dir_projected = "./experiments/Encodec-Seanet-lookup/Thu-Jul-25-13_51_42-2024"

output_path = "./ablation_result/codebook_utilization"

def Get_Codebook_Entries_Utilization(ecg_paths, model, device="cuda:0"):
    global bins
    vq_entries_counter = [0] * bins
    model = model.to(device)

    for path in tqdm(ecg_paths):
        ecg0 = np.load(path, allow_pickle=True).astype("float32").reshape(1,1,-1)
        ecg0 = torch.tensor(ecg0).to(device)

        indices = model.compress_to_indices(ecg0)
        indices = [x.to('cpu').detach().numpy().reshape(-1) for x in indices]
        indices = np.concatenate(indices)

        unique_values, counts = np.unique(indices, return_counts=True)
        for value, count in zip(unique_values, counts):
            vq_entries_counter[value - 1] += count

    utilized_entries = len([x for x in vq_entries_counter if x != 0])
    print(f"Codebook utilization:{utilized_entries / len(vq_entries_counter)}")

    return utilized_entries / len(vq_entries_counter)


# print("====== Searching Best Checkpoints ======")
# # Get best epoch's checkpoints
# for file in tqdm(rst_files):
#
#     df_rst = pd.read_csv(file)
#
#     CR = int(df_rst['CR'].values[0])
#     e_dims = int(df_rst['e_dims'].values[0])
#     bins = int(df_rst['codebook_size'].values[0])
#     hop_length = int(df_rst["hop_length"].values[0])
#     bs = int(df_rst['batch_size'].values[0])
#     nq = int(df_rst['n_q'].values[0])
#
#     best_epoch_idx = df_rst['PRDN'].idxmin()
#     best_epoch = int(df_rst.loc[best_epoch_idx]['epoch'])
#
#     best_ckpt_path = osp.join(src_ckpt_dir,
#                               f"dims{e_dims}_bins{bins}_nq{nq}",
#                               "checkpoints",
#                               f"{best_epoch:08}.pth")
#
#     destination_folder = osp.join(output_path,
#                                   "projected_False",
#                                 f"{CR}-{e_dims}-{bins}-{nq}")
#
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     destination_path = osp.join(destination_folder, f"{best_epoch:08}.pth")
#
#     shutil.copy2(best_ckpt_path, destination_path)
#
# # Get best epoch's checkpoints for models with codebook projection
# for file in tqdm(rst_files_projected):
#
#     df_rst = pd.read_csv(file)
#
#     bins = df_rst['codebook_size'].values[0]
#     if np.isnan(bins):
#         continue
#
#     bins = int(bins)
#     CR = int(df_rst['CR'].values[1])
#     e_dims = int(df_rst['e_dims'].values[0])
#     codebook_dims = int(df_rst['codebook_dims'].values[0])
#
#     nq = int(df_rst['n_q'].values[0])
#
#     best_epoch_idx = df_rst['PRDN'].idxmin()
#     best_epoch = int(df_rst.loc[best_epoch_idx]['epoch'])
#
#     best_ckpt_path = osp.join(src_ckpt_dir_projected,
#                               f"dims{e_dims}_cbdims{codebook_dims}_bins{bins}_nq{nq}",
#                               "checkpoints",
#                               f"{best_epoch:08}.pth")
#
#     destination_folder = osp.join(output_path,
#                                   "projected_True",
#                                   f"{CR}-{e_dims}-{codebook_dims}-{bins}-{nq}")
#
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     destination_path = osp.join(destination_folder, f"{best_epoch:08}.pth")
#
#     shutil.copy2(best_ckpt_path, destination_path)


print("====== Getting All Results ======")
result_dirs = glob("./ablation_result/codebook_utilization/*/*")
ratios = [8,5,4,2]
ecg_paths = glob("./data/full_data/mit-bih/*/*/*.npy")

utilization_dict = {}

for rst_dir in tqdm(result_dirs):
    ckpt_pth = glob(osp.join(rst_dir, "*.pth"))[0]
    projection = rst_dir.split('\\')[-2]
    projection = eval(projection.split("_")[-1])
    exp_info = rst_dir.split('\\')[-1]
    if projection:
        CR, e_dims, codebook_dims, bins, nq = [int(x) for x in exp_info.split('-')]
    else:
        CR, e_dims, bins, nq = [int(x) for x in exp_info.split('-')]
        codebook_dims = None

    model = EncodecModel.get_exp_model(ratios=ratios,
                                       e_dims=e_dims,
                                       codebook_dims=codebook_dims,
                                       codebook_size=bins,
                                       n_q=nq,
                                       use_lookup=projection)
    model.load_state_dict(torch.load(ckpt_pth))

    utilization = Get_Codebook_Entries_Utilization(ecg_paths, model, device="cuda:0")
    utilization_dict[f"Projection{projection}_size{bins}"] = utilization
    # Get_All_Result(model, rst_dir, CR, device="cuda:0")

import json
with open("./ablation_result/codebook_utilization/utilization_rate.json", "w") as json_file:
    json.dump(utilization_dict, json_file)

# print("====== Getting Final Result ======")
# result_dirs = glob("./ablation_result/codebook_utilization/*/*")
#
# with open(osp.join(output_path, "result.csv"), 'w', encoding="utf-8") as f:
#     csv_writer = csv.writer(f)
#     head = ["projection", "e_dims", "codebook_dims", "bins", "nq", "CR", "rmse", "prd", "prdn", "QS"]
#     csv_writer.writerow(head)
#
#     for rst_dir in tqdm(result_dirs):
#         projection = rst_dir.split('\\')[-2]
#         projection = eval(projection.split("_")[-1])
#         exp_info = rst_dir.split('\\')[-1]
#         if projection:
#             CR, e_dims, codebook_dims, bins, nq = [int(x) for x in exp_info.split('-')]
#         else:
#             CR, e_dims, bins, nq = [int(x) for x in exp_info.split('-')]
#             codebook_dims = None
#
#         rst_pth = glob(osp.join(rst_dir, "*.csv"))[0]
#         with open(rst_pth, 'r', encoding="utf-8") as rf:
#             df_rst = pd.read_csv(rf)
#             mean_rst = df_rst[['rms', 'prd', 'prdn','QS']].mean()
#
#             values = [projection, e_dims, codebook_dims, bins, nq,
#                       CR, mean_rst['rms'], mean_rst['prd'], mean_rst['prdn'], mean_rst['QS']]
#
#             csv_writer.writerow(values)
#
#     f.close()


