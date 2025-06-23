import h5py
import numpy as np
import torch

def read_matlab_data(mat_file):
    with h5py.File(mat_file, 'r') as f:
        data = f['data']
        labels = f['labels']
        n_samples = data.shape[0]
        K = data.shape[1]
        print(f"Number of samples: {n_samples}, Number of frames per trajectory: {K}")

        # Parse labels
        label_list = []
        for i in range(n_samples):
            traj_labels = []
            for k in range(K):
                ref = labels[i, k]         # Remove [0]
                traj_labels.append(np.array(f[ref]).squeeze())
            label_list.append(traj_labels)
        label_arr = np.array(label_list, dtype=object)

        # Parse data
        data_list = []
        for i in range(n_samples):
            traj_data = []
            for k in range(K):
                ref = data[i, k]           # Remove [0]
                traj_data.append(np.array(f[ref]).astype(np.float32))
            data_list.append(traj_data)
        data_arr = np.array(data_list, dtype=object)
    return data_arr, label_arr

def split_and_save(data_arr, label_arr, valid_ratio=0.1):
    n_samples = data_arr.shape[0]
    n_valid = int(n_samples * valid_ratio)
    idxs = np.random.permutation(n_samples)
    train_idxs = idxs[:-n_valid]
    valid_idxs = idxs[-n_valid:]

    # Save in the same format and filenames as the original script
    torch.save(data_arr[train_idxs], 'train_data.pt')
    torch.save(label_arr[train_idxs], 'train_label.pt')
    torch.save(data_arr[valid_idxs], 'valid_data.pt')
    torch.save(label_arr[valid_idxs], 'valid_label.pt')
    print(f'Training set: {len(train_idxs)}, Validation set: {len(valid_idxs)}. Filenames match the original script.')

if __name__ == "__main__":
    np.random.seed(2024)  # Optional: Set random seed for reproducible splits
    mat_file = "/home/zktx/YSX/YSX_TBD/simulate_data/test_data_phys.mat"
    data_arr, label_arr = read_matlab_data(mat_file)
    split_and_save(data_arr, label_arr, valid_ratio=0.1)