import json
import os
import os.path as osp
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

# configuration of training
test_fraction = 0.1
val_fraction = 0.1

extension = "npy"
mit_data_path = osp.abspath("../data/train_data/*/*/*/*.{}".format(extension))

dataset_config = [('mit', mit_data_path)]

output_path = "../train_config"
os.makedirs(output_path, exist_ok=True)

random_state = 7

name_list = []
temp_name = ''


def generate_dataset(path, data_type):
    dataset = []

    for file in tqdm(glob(path)):
        # if data_type == 'mit':
        filename = file.split("\\")[-1]
        name, lead, _ = filename.split("_")
        peak_file = f"D:\\AI\\Compression\\ECG-Compress\\ecg_classifier\\peak_label\\peaks{name}.npy"

        full_file = f"../data/full_data/mit-bih/{name}/{lead}/{name}_{lead}_seg0.npy"
        full_ecg = np.load(full_file).astype("float32")
        scale = np.max(np.abs(full_ecg))
        offset = np.min(full_ecg)

        dataset.append(
            {
                "name": name,
                "lead": lead,
                "filename": osp.splitext(filename)[0],
                "path": file,
                "peaks_path": peak_file,
                "scale": str(scale),
                "offset": str(offset)
            },
        )

    return dataset


if __name__ == "__main__":
    dataset = []
    mix_dataset = []
    mix_slice_dataset = []

    for item in dataset_config:
        dataset_type, data_path = item
        dataset = generate_dataset(data_path, dataset_type)

        data = pd.DataFrame(dataset)

        # get test set
        test_ids = []
        test_ids.extend(
            data.sample(frac=test_fraction, random_state=random_state)
            .index,
        )
        test = data.loc[test_ids, :]
        train_data = data[~data.index.isin(test.index)]

        # get validation set
        val_ids = []
        val_ids.extend(
            train_data.sample(frac=val_fraction, random_state=random_state)
            .index,
        )

        val = train_data.loc[val_ids, :]

        # remove validation set
        train = train_data[~train_data.index.isin(val.index)]

        os.makedirs(osp.join(output_path, dataset_type), exist_ok=True)

        train.to_json(osp.join(output_path, dataset_type, "train.json"), orient="records")
        val.to_json(osp.join(output_path, dataset_type, "val.json"), orient="records")
        test.to_json(osp.join(output_path, dataset_type, "test.json"), orient="records")

