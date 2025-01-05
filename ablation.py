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
from models.model import ECGCodec
import torch


rst_files = glob("./experiments/Encodec-Seanet/Fri-Jul-19-15_36_01-2024/*/result/*")
rst_files_projected = glob("./experiments/Encodec-Seanet-lookup/Sat-Jul-20-21_38_24-2024/*/result/*")

src_ckpt_dir = "./experiments/Encodec-Seanet/Fri-Jul-19-15_36_01-2024"
src_ckpt_dir_projected = "./experiments/Encodec-Seanet-lookup/Sat-Jul-20-21_38_24-2024"

output_path = "./ablation_result"


print("====== Searching Best Checkpoints ======")
# Get best epoch's checkpoints
for file in tqdm(rst_files):

    df_rst = pd.read_csv(file)

    CR = int(df_rst['CompressRate'].values[0])
    e_dims = int(df_rst['e_dims'].values[0])
    bins = int(df_rst['codebook_size'].values[0])
    hop_length = int(df_rst["hop_length"].values[0])
    bs = int(df_rst['batch_size'].values[0])
    nq = int(df_rst['n_q'].values[0])

    best_epoch_idx = df_rst['PRDN'].idxmin()
    best_epoch = int(df_rst.loc[best_epoch_idx]['epoch'])

    best_ckpt_path = osp.join(src_ckpt_dir,
                              f"bs_{bs}_dims{e_dims}_hop{hop_length}_bins{bins}_nq{nq}_ecgFalse",
                              "checkpoints",
                              f"{best_epoch:08}.pth")

    destination_folder = osp.join(output_path,
                                "projected_False",
                                f"{CR}-{e_dims}-{bins}-{nq}")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    destination_path = osp.join(destination_folder, f"{best_epoch:08}.pth")

    shutil.copy2(best_ckpt_path, destination_path)
#
# # Get best epoch's checkpoints for models with codebook projection
# for file in tqdm(rst_files_projected):
#
#     df_rst = pd.read_csv(file)
#
#     bins = df_rst['codebook_size'].values[0]
#     if np.isnan(bins) or int(bins) != 1024:
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
#
#
# print("====== Getting All Results ======")
# result_dirs = glob("./ablation_result/*/*")
# ratios = [8,5,4,2]
#
# for rst_dir in tqdm(result_dirs):
#     ckpt_pth = glob(osp.join(rst_dir, "*.pth"))[0]
#     projection = rst_dir.split('\\')[-2]
#     projection = eval(projection.split("_")[-1])
#     exp_info = rst_dir.split('\\')[-1]
#     if projection:
#         CR, e_dims, codebook_dims, bins, nq = [int(x) for x in exp_info.split('-')]
#     else:
#         CR, e_dims, bins, nq = [int(x) for x in exp_info.split('-')]
#         codebook_dims = None
#
#     model = ECGCodec.get_exp_model(ratios=ratios,
#                                        e_dims=e_dims,
#                                        codebook_dims=codebook_dims,
#                                        codebook_size=bins,
#                                        n_q=nq,
#                                        use_lookup=projection)
#     model.load_state_dict(torch.load(ckpt_pth))
#
#     Get_All_Result(model, rst_dir, CR, device="cuda:0")

print("====== Getting Final Result ======")
result_dirs = glob("./ablation_result/*/*")

with open(osp.join(output_path, "result.csv"), 'w', encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    head = ["projection", "e_dims", "codebook_dims", "bins", "nq", "CR", "rmse", "prd", "prdn", "QS"]
    csv_writer.writerow(head)

    for rst_dir in tqdm(result_dirs):
        projection = rst_dir.split('\\')[-2]
        projection = eval(projection.split("_")[-1])
        exp_info = rst_dir.split('\\')[-1]
        if projection:
            CR, e_dims, codebook_dims, bins, nq = [int(x) for x in exp_info.split('-')]
        else:
            CR, e_dims, bins, nq = [int(x) for x in exp_info.split('-')]
            codebook_dims = None

        rst_pth = glob(osp.join(rst_dir, "*.csv"))[0]
        with open(rst_pth, 'r', encoding="utf-8") as rf:
            df_rst = pd.read_csv(rf)
            mean_rst = df_rst[['rms', 'prd', 'prdn','QS']].mean()

            values = [projection, e_dims, codebook_dims, bins, nq,
                      CR, mean_rst['rms'], mean_rst['prd'], mean_rst['prdn'], mean_rst['QS']]

            csv_writer.writerow(values)

    f.close()


