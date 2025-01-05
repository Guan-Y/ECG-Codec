import json
import numpy as np
import wfdb
from scipy.signal import find_peaks
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, Dataset
import torch


def my_collate_fn(batch):
    # 提取每个样本中的信号数据和峰值数据
    sigs = [sample["sig"] for sample in batch]
    peaks = [sample["peaks"] for sample in batch]

    # 转换成张量并返回
    return {"sig": torch.FloatTensor(sigs), "peaks": peaks}


class ECGDataset(Dataset):
    def __init__(self, traindata_path):
        super().__init__()
        self.data = json.load(open(traindata_path))

    def __getitem__(self, index):
        sig = np.load(self.data[index]["path"], allow_pickle=True).astype("float32")
        # peaks = np.load(self.data[index]["peaks_path"]).astype(int)

        sig = sig.reshape(1, sig.shape[0])
        # peaks = peaks.reshape(1, peaks.shape[0])
        scale = float(self.data[index]["scale"])
        offset = float(self.data[index]["offset"])

        return {
            "sig": sig,
            "peaks": self.data[index]["peaks_path"],
            "scale": scale,
            "offset": offset
        }

    def get_dataloader(self, num_workers=4, batch_size=16, shuffle=False):
        data_loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        return data_loader

    def __len__(self):
        return len(self.data)


def callback_get_label(dataset, idx):
    return dataset[idx]["class"]
