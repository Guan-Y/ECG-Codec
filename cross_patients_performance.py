from benchmark import Get_Recons_Data
from benchmark import Obtain_Accuracy
from benchmark import Obtain_Common_Metrics
from benchmark import Obtain_Test_Requirements
from glob import glob
import math
import torch
from models.model import EncodecModel
import csv
import numpy as np
import os.path as osp
from tqdm import tqdm

def Get_All_Result(model, rst_dir, CR, device="cpu"):
    files = glob("./data/full_data/mit-bih/*/*/*.npy")
    # ecg_model, test_subject, mode, peaks, labels, device = Obtain_Test_Requirements()
    model = model.to(device)

    # create new result file
    rst_filename = osp.join(rst_dir, f'all_result_{CR}.csv')
    with open(rst_filename, "w", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        head = ["patient_id", "rms", "prd", "prdn", "snr", "QS", "accuracy"]
        csv_writer.writerow(head)

        for file in tqdm(files):
            patient_id = file.split('\\')[-3]
            ecg_input = np.load(file, allow_pickle=True).astype("float32")

            ecg_input, ecg_reconstruct, scale, offset \
                = Get_Recons_Data(model, ecg_input, device=device)

            rms, prd, prdn, snr \
                = Obtain_Common_Metrics(ecg_input, ecg_reconstruct, scale, offset)

            QS = CR / (prd * 100)

            # accuracy = Obtain_Accuracy(ecg_model, ecg_input, mode, peaks, labels, device)
            accuracy = 0.5

            value = [patient_id, rms, prd, prdn, snr, QS, accuracy]
            csv_writer.writerow(value)
    f.close()


if __name__ == "__main__":
    bs = 16
    nq = 4
    bins = 1024
    dim = 4
    codebook_dims = 16
    ratios = [8, 5, 4, 2]
    hop_length = 1
    for item in ratios:
        hop_length *= item

    # CR = hop_length * 11 / (nq * math.log2(bins))
    CR = hop_length * 11 / (32 * dim)

    exp_time = "Tue-Jul-23-10_28_51-2024"

    # model_path = f"./experiments/Encodec-Seanet/{exp_time}/" \
    #            f"bs_{bs}_dims{dim}_hop{hop_length}_bins{bins}_nq{nq}_ecgFalse/" \
    #            f"checkpoints/00000199.pth"

    model_path = f"./experiments/Encodec-Seanet-lookup/{exp_time}/" \
                 f"dims{dim}_cbdims16_bins256_nq4/" \
                 f"checkpoints/00000199.pth"

    test_model = EncodecModel.get_exp_model(ratios, dim, codebook_dims, bins, nq, use_lookup=True)
    test_model.load_state_dict(torch.load(model_path))

    rst_dir = "./experiments"
    Get_All_Result(test_model, rst_dir, CR, device="cuda:0")
