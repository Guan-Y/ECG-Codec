import torch
import numpy as np
import csv
import time
import json
import os.path as osp
from glob import glob
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


src_data_path = "./data/full_data/mit-bih"
peak_label_path = "./ecg_classifier/peak_label"
class_mapper_path = "./ecg_classifier/class-mapper.json"
pretrained_path = glob("./ecg_classifier/ecg_checkpoints/heartnetIEEE/*.pth")[0]
physio_data_path = "./data/full_data/physionet-2017"


class EcgReconsDataset(Dataset):
    def __init__(self, inputs, labels, mapping_path):
        super().__init__()
        self.inputs = inputs
        self.labels = labels
        self.mapper = json.load(open(mapping_path))

    def __getitem__(self, index):
        sig = self.inputs[index]
        sig = sig.reshape(1, sig.shape[0])

        return {"sig": sig, "class": self.mapper[self.labels[index]]}

    def get_dataloader(self, num_workers=4, batch_size=1, shuffle=False):
        data_loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
        return data_loader

    def __len__(self):
        return len(self.inputs)

def Save_Result_CSV(rst_dir, info_dict):
    # save into csv
    filename = osp.join(rst_dir, 'exp_result.csv')
    with open(filename, "a", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        if osp.getsize(filename) == 0:
            head = ['test_time',
                    'dataset',
                    'optimizer',
                    'e_dims',
                    'codebook_dims',
                    'codebook_size',
                    'n_q',
                    'batch_size',
                    'learning_rate',
                    'epoch',
                    'test_loss',
                    'hop_length',
                    'CR',
                    'RMS',
                    'PRD',
                    'PRDN',
                    'SNR',
                    'QS',
                    'src_accuracy',
                    'recons_accuracy']
            csv_writer.writerow(head)

        value = [info_dict["test_time"],
                 info_dict["dataset"],
                 info_dict["optimizer"],
                 info_dict['e_dims'],
                 info_dict['codebook_dims'],
                 info_dict['codebook_size'],
                 info_dict['n_q'],
                 info_dict["batch_size"],
                 info_dict["lr"],
                 info_dict["epoch"],
                 info_dict["test_loss"],
                 info_dict["hop_length"],
                 info_dict["CR"],
                 info_dict["RMS"],
                 info_dict["PRD"],
                 info_dict["PRDN"],
                 info_dict["SNR"],
                 info_dict["QS"],
                 info_dict["src_accuracy"],
                 info_dict["recons_accuracy"]
                 ]
        csv_writer.writerow(value)


def Peak2ECG(ecg, peaks, labels, mode):
    mask_left = (peaks - mode // 2) > 0
    mask_right = (peaks + mode // 2) < len(ecg)

    classes = ["N", "V", "slash", "R", "L", "A", "!", "E"]
    mask_labels = [(label in classes) for label in labels]

    mask = mask_left & mask_right & mask_labels

    temp_peaks = peaks[mask]
    temp_labels = labels[mask]
    ecgs = []

    for peak in temp_peaks:
        left, right = peak - mode // 2, peak + mode // 2
        ecgs.append(ecg[left:right])

    return ecgs, temp_labels


def Obtain_Common_Metrics(src, recons, scale, offset):
    src_len = len(src)

    # RMS
    temp = np.sum(np.square(src - recons))  # origin - reconstruct
    identity = np.sum(np.square(src))
    rms = np.sqrt((temp / src_len))

    # PRD(%)
    prd = np.sqrt(temp / identity)

    # PRDN(%)
    Gm = np.mean(src)
    temp_mean = np.square(src - Gm)  # normalized
    prdn = np.sqrt(temp / np.sum(temp_mean))

    # SNR(dB)
    signal_power = np.mean(src ** 2)
    noise = recons - src
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    return rms, prd, prdn, snr


def Get_Recons_Data(model, ecg_input, device="cuda:0"):
    offset = np.min(ecg_input)
    ecg_input -= offset
    scale = np.max(np.abs(ecg_input))
    ecg_input = ecg_input / scale

    src_len = len(ecg_input)
    ecg_input = ecg_input.reshape(1, 1, -1)
    ecg_input = torch.tensor(ecg_input)
    ecg_input = ecg_input.to(device)

    time_start = time.time()
    ecg_compressed = model.compress(ecg_input)
    time_end = time.time()
    print(f"{device} Encode time: {time_end - time_start} (s)")

    time_start = time.time()
    ecg_reconstruct = model.decompress(ecg_compressed)
    time_end = time.time()
    print(f"{device} Decode time: {time_end - time_start} (s)")

    ecg_reconstruct = ecg_reconstruct[:, :, :src_len]

    # move origin data & reconstruct to cpu for analysis
    ecg_input = ecg_input.to("cpu")
    ecg_input = ecg_input.detach().numpy().reshape(-1)
    ecg_reconstruct = ecg_reconstruct.to("cpu")
    ecg_reconstruct = ecg_reconstruct.detach().numpy().reshape(-1)

    return ecg_input, ecg_reconstruct, scale, offset


def Result_Analysis(model, info, rst_dir):

    src_files = "./data/full_data/mit-bih/100/MLII/100_MLII_seg0.npy"

    device = "cuda:0"

    model = model.to(device)

    ecg_input = np.load(src_files, allow_pickle=True).astype("float32")

    ecg_input, ecg_reconstruct, scale, offset = Get_Recons_Data(model, ecg_input, device)

    rms, prd, prdn, snr = Obtain_Common_Metrics(ecg_input, ecg_reconstruct, scale, offset)

    test_time = str(time.asctime().replace(':', '_'))
    test_time = test_time.replace(' ', '-')

    info["RMS"] = rms
    info["PRD"] = prd
    info["PRDN"] = prdn
    info["SNR"] = snr
    info["QS"] = info["CR"] / (prd*100)
    info["test_time"] = test_time

    compress_rate = info["CR"]

    # show infomation
    print("---------- Encodec Test Result ----------")
    print(f"CR: {compress_rate}")
    print(f"RMS: {rms:.4f}")
    print(f"PRD(%):{prd * 100:.2f} %")
    print(f"PRDN(%): {prdn * 100:.2f} %")
    print(f"SNR: {snr:.2f}dB")
    print(f"QS(Quality Score): {compress_rate / (prd * 100):.2f}")

    Save_Result_CSV(rst_dir, info)

    return ecg_input, ecg_reconstruct


def Plot(src, recons):
    segment_len = 108000
    fig, ax = plt.subplots()
    plt.rcParams['font.family'] = 'Times New Roman'
    t = np.linspace(0, segment_len, segment_len)
    ax.plot(t, src[0:segment_len], color='blue', label='origin signal')
    ax.plot(t, recons[0:segment_len],color='orange', label='reconstruct signal')
    ax.set_title('ECG Codec (CR=88)', fontsize=14, fontname='Times New Roman', weight='bold')
    ax.legend()
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    import math
    bs = 16
    nq = 8
    bins = 1024
    e_dims = 1024
    codebook_dims = 16
    ratios = [8, 5, 4, 2]
    hop_length = 1
    for item in ratios:
        hop_length *= item

    CR = 88

    rvq_path = f"./checkpoints/" \
                 f"CR{CR}_edims{e_dims}_cbdims{codebook_dims}.pth/" \
                 f"checkpoints/00000199.pth"

    rst_dir = "./experiments/temp"

    from models.model import EncodecModel
    test_model = EncodecModel.get_exp_model(ratios, e_dims, codebook_dims, bins, nq, use_lookup=True)
    test_model.load_state_dict(torch.load(rvq_path))
    ecg_input, ecg_reconstruct = Result_Analysis(test_model, train_param, rst_dir)

    Plot(ecg_input, ecg_reconstruct)
