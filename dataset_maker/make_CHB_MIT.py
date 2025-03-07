# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import os
import pickle

from multiprocessing import Pool
import numpy as np
import mne

chOrder_standard = [ "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]

standard_channels = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8" , "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                ch_names = raw.ch_names
                if "T8-P8-0" in ch_names and "T8-P8-1" in ch_names:
                    raw.drop_channels(["T8-P8-1"])
                    raw.rename_channels({"T8-P8-0": "T8-P8"})
    
                channels_to_drop = [ch for ch in raw.ch_names if ch not in standard_channels]

                if channels_to_drop:
                    raw.drop_channels(channels_to_drop)
                    print(raw.ch_names)
                if len(raw.ch_names) == len(chOrder_standard):
                    raw.reorder_channels(chOrder_standard)
                else:
                    raise Exception(f"Channel count mismatch. Found {len(raw.ch_names)}, expected {len(chOrder_standard)}")
                
                raw.filter(l_freq=0.1, h_freq=75.0)
                raw.notch_filter(50.0)
                raw.resample(200, n_jobs=5)
            
                ch_name = raw.ch_names
                raw_data = raw.get_data(units='uV')
                channeled_data = raw_data.copy()
                for i in range(channeled_data.shape[1] // 2000):
                    dump_path = os.path.join(
                        dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                    )
                    pickle.dump(
                        {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                        open(dump_path, "wb"),
                    )
                    
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                with open("chb-mit-process-error-files.txt", "a") as f:
                    f.write(f"{file}: {str(e)}\n")
                continue


if __name__ == "__main__":
    # root to abnormal dataset
    root = ""
    channel_std = "01_tcp_ar1"

    # train, val abnormal subjects
    train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    train_val_a_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_abnormal)])
    )
    np.random.shuffle(train_val_a_sub)
    train_a_sub, val_a_sub = (
        train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
        train_val_a_sub[int(len(train_val_a_sub) * 0.8) :],
    )

    # train, val normal subjects
    train_val_normal = os.path.join(root, "train", "normal", channel_std)
    train_val_n_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_normal)])
    )
    np.random.shuffle(train_val_n_sub)
    train_n_sub, val_n_sub = (
        train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
        train_val_n_sub[int(len(train_val_n_sub) * 0.8) :],
    )

    # test abnormal subjects
    test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    # test normal subjects
    test_normal = os.path.join(root, "eval", "normal", channel_std)
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, "processed")):
        os.makedirs(os.path.join(root, "processed"))

    if not os.path.exists(os.path.join(root, "processed", "train")):
        os.makedirs(os.path.join(root, "processed", "train"))
    train_dump_folder = os.path.join(root, "processed", "train")

    if not os.path.exists(os.path.join(root, "processed", "val")):
        os.makedirs(os.path.join(root, "processed", "val"))
    val_dump_folder = os.path.join(root, "processed", "val")

    if not os.path.exists(os.path.join(root, "processed", "test")):
        os.makedirs(os.path.join(root, "processed", "test"))
    test_dump_folder = os.path.join(root, "processed", "test")

    # fetch_folder, sub, dump_folder, labels
    parameters = []
    for train_sub in train_a_sub:
        parameters.append([train_val_abnormal, train_sub, train_dump_folder, 1])
    for train_sub in train_n_sub:
        parameters.append([train_val_normal, train_sub, train_dump_folder, 0])
    for val_sub in val_a_sub:
        parameters.append([train_val_abnormal, val_sub, val_dump_folder, 1])
    for val_sub in val_n_sub:
        parameters.append([train_val_normal, val_sub, val_dump_folder, 0])
    for test_sub in test_a_sub:
        parameters.append([test_abnormal, test_sub, test_dump_folder, 1])
    for test_sub in test_n_sub:
        parameters.append([test_normal, test_sub, test_dump_folder, 0])

    # split and dump in parallel
    with Pool(processes=20) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)