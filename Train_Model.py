#Train_Model.py V1
# import torch
# import numpy as np
# import os
# from torch.utils.data import TensorDataset, DataLoader
# from unet_model import UNet
# from train_parts import train_model

# def to_numeric_array(x, dtype=np.float32):
#     if isinstance(x, np.ndarray) and x.dtype != object:
#         return x.astype(dtype)
#     seq = list(x) if (isinstance(x, (list, tuple)) or
#                      (isinstance(x, np.ndarray) and x.dtype == object)) else None
#     if seq is not None:
#         arrs = [np.array(elem, dtype=dtype) for elem in seq]
#         return np.stack(arrs, axis=0)
#     raise TypeError(f"Cannot convert type {type(x)} to numeric array")

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load raw pickled data
#     train_data_np   = torch.load('train_data.pt', weights_only=False)
#     valid_data_np   = torch.load('valid_data.pt', weights_only=False)
#     train_label_np  = torch.load('train_label.pt', weights_only=False)
#     valid_label_np  = torch.load('valid_label.pt', weights_only=False)

#     # Convert labels and data to numeric
#     train_label_np = to_numeric_array(train_label_np, dtype=np.float32)
#     valid_label_np = to_numeric_array(valid_label_np, dtype=np.float32)
#     train_data_np  = to_numeric_array(train_data_np,  dtype=np.float32)
#     valid_data_np  = to_numeric_array(valid_data_np,  dtype=np.float32)

#     # Map physical [r,v] to pixel indices
#     N, K, H, W = train_data_np.shape
#     R_min, R_max = 1000.0, 2500.0
#     V_max = 300.0
#     tl = train_label_np[:, -1, :]
#     vl = valid_label_np[:, -1, :]

#     dr = (R_max - R_min) / (W - 1)
#     dv = (2 * V_max) / (H - 1)

#     r_idx  = ((tl[:,0] - R_min)/dr).round().clip(0, W-1).astype(np.int64)
#     d_idx  = ((tl[:,1] + V_max)/dv).round().clip(0, H-1).astype(np.int64)
#     vr_idx = ((vl[:,0] - R_min)/dr).round().clip(0, W-1).astype(np.int64)
#     vd_idx = ((vl[:,1] + V_max)/dv).round().clip(0, H-1).astype(np.int64)

#     train_label_np = np.stack([r_idx, d_idx], axis=1)
#     valid_label_np = np.stack([vr_idx, vd_idx], axis=1)
#     # # right after stacking and before wrapping into tensors
#     # print("Example pixel©\coords (r_idx,d_idx):", train_label_np[:10])
#     # print("Max pixel©\coord:", train_label_np.max(axis=0), "should be < (W,H)=({}, {})".format(W, H))


#     # Create tensors and move to device
#     train_data_t   = torch.from_numpy(train_data_np).to(device)
#     valid_data_t   = torch.from_numpy(valid_data_np).to(device)
#     train_label_t  = torch.from_numpy(train_label_np).to(device)
#     valid_label_t  = torch.from_numpy(valid_label_np).to(device)

#     # # normalize per-sample: (batch, K, H, W)
#     # mean = train_data_t.mean(dim=[2,3], keepdim=True)
#     # std  = train_data_t.std(dim=[2,3], keepdim=True) + 1e-6
#     # train_data_t = (train_data_t - mean) / std

#     # mean = valid_data_t.mean(dim=[2,3], keepdim=True)
#     # std  = valid_data_t.std(dim=[2,3], keepdim=True) + 1e-6
#     # valid_data_t = (valid_data_t - mean) / std


#     # DataLoaders
#     batch_size = 32
#     train_ds = TensorDataset(train_data_t, train_label_t)
#     valid_ds = TensorDataset(valid_data_t, valid_label_t)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     valid_loader = DataLoader(valid_ds, batch_size=len(valid_ds), shuffle=False)

#     # 5. Figure out H and W from your data
#     in_ch = train_data_t.shape[1]        # number of RD ¡°frames¡±
#     H     = train_data_t.shape[2]        # height of each RD image
#     W     = train_data_t.shape[3]        # width  of each RD image

# # 6. Instantiate your UNet to output H*W logits
# #    (make sure your unet_model.UNet signature now accepts H and W!)
#     emis_DNN = UNet(
#         in_channels=in_ch,
#         base_channels=64,
#         bilinear=True,
#         H=H,
#         W=W
# ).to(device)


#     #Train_Model.py V2
#     # Training
#     train_losses, train_accs, valid_losses, valid_accs = train_model(
#         emis_DNN,
#         train_loader=train_loader,
#         valid_loader=valid_loader,
#         learning_rate=1e-3,
#         epochs=75,
#         weight_decay=1
#     )

#     # Print final epoch metrics
#     print(f"Train loss: {train_losses[-1]:.4f}, Train acc: {train_accs[-1]:.2%}")
#     print(f"Val   loss: {valid_losses[-1]:.4f}, Val   acc: {valid_accs[-1]:.2%}")

#     # Save model
#     torch.save(emis_DNN.state_dict(), 'Emission_DNN/dnn_state')

# #Train_Model.py V2

# import os
# import torch
# import numpy as np
# from torch.utils.data import TensorDataset, DataLoader
# from unet_model import UNet
# from train_parts import train_model

# def to_numeric_array(x, dtype=np.float32):
#     if isinstance(x, np.ndarray) and x.dtype != object:
#         return x.astype(dtype)
#     seq = list(x) if (isinstance(x, (list, tuple)) or
#                      (isinstance(x, np.ndarray) and x.dtype == object)) else None
#     if seq is not None:
#         arrs = [np.array(elem, dtype=dtype) for elem in seq]
#         return np.stack(arrs, axis=0)
#     raise TypeError(f"Cannot convert type {type(x)} to numeric array")

# def make_gaussian_heatmap_array(coords: np.ndarray, H: int, W: int, sigma: float=3.0):
#     """
#     coords: (N,2) discrete pixel indices [r_idx, d_idx]
#     Returns: (N,1,H,W) normalized Gaussian heatmaps.
#     """
#     N = coords.shape[0]
#     ys = np.arange(H)[:, None]
#     xs = np.arange(W)[None, :]
#     maps = np.zeros((N, 1, H, W), dtype=np.float32)

#     for i, (r, d) in enumerate(coords):
#         assert 0 <= r < W and 0 <= d < H, f"Out-of-bounds index r={r},d={d}"
#         g    = np.exp(-((xs - r)**2 + (ys - d)**2) / (2 * sigma**2))
#         m    = g.max()
#         maps[i, 0] = g/m if m>0 else g
#     return maps

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 1) Load raw data
#     train_data_raw  = torch.load('train_data.pt',   weights_only=False)
#     valid_data_raw  = torch.load('valid_data.pt',   weights_only=False)
#     train_label_raw = torch.load('train_label.pt',  weights_only=False)
#     valid_label_raw = torch.load('valid_label.pt',  weights_only=False)

#     # 2) Collapse to last-frame physical coords and convert to numpy
#     train_label_phys = to_numeric_array(train_label_raw, dtype=np.float32)[:, -1, :]
#     valid_label_phys = to_numeric_array(valid_label_raw, dtype=np.float32)[:, -1, :]
#     train_data_np    = to_numeric_array(train_data_raw,  dtype=np.float32)
#     valid_data_np    = to_numeric_array(valid_data_raw,  dtype=np.float32)

#     # 3) Discretize physical ¡ú pixel indices
#     N, K, H, W = train_data_np.shape
#     R_min, R_max = 1000.0, 2500.0   # [m]
#     V_max        = 300.0            # [m/s]
#     dr = (R_max - R_min) / (W - 1)
#     dv = (2*V_max)       / (H - 1)

#     r_idx_train = np.clip(np.round((train_label_phys[:,0]-R_min)/dr), 0, W-1).astype(int)
#     d_idx_train = np.clip(np.round((train_label_phys[:,1]+V_max)/dv),   0, H-1).astype(int)
#     r_idx_valid = np.clip(np.round((valid_label_phys[:,0]-R_min)/dr), 0, W-1).astype(int)
#     d_idx_valid = np.clip(np.round((valid_label_phys[:,1]+V_max)/dv),   0, H-1).astype(int)

#     train_coords = np.stack([r_idx_train, d_idx_train], axis=1)
#     valid_coords = np.stack([r_idx_valid, d_idx_valid], axis=1)

#     # 4) Build heatmaps with sigma=3.0
#     train_heatmap_np = make_gaussian_heatmap_array(train_coords, H, W, sigma=3.0)
#     valid_heatmap_np = make_gaussian_heatmap_array(valid_coords, H, W, sigma=3.0)

#     # 5) To tensors & device
#     train_data_t    = torch.from_numpy(train_data_np).to(device)
#     train_heatmap_t = torch.from_numpy(train_heatmap_np).to(device)
#     valid_data_t    = torch.from_numpy(valid_data_np).to(device)
#     valid_heatmap_t = torch.from_numpy(valid_heatmap_np).to(device)

#     # 6) Per-sample normalization
#     for t in [train_data_t, valid_data_t]:
#         m = t.mean(dim=[2,3], keepdim=True)
#         s = t.std(dim=[2,3], keepdim=True) + 1e-6
#         t.sub_(m).div_(s)

#     # 7) DataLoaders
#     bs = 32
#     train_loader = DataLoader(TensorDataset(train_data_t, train_heatmap_t),
#                               batch_size=bs, shuffle=True)
#     valid_loader = DataLoader(TensorDataset(valid_data_t, valid_heatmap_t),
#                               batch_size=len(valid_data_t), shuffle=False)

#     # 8) UNet instantiation
#     model = UNet(in_channels=K, base_channels=64, bilinear=True, H=H, W=W)
#     model.to(device)

#     # 9) Optional resume
#     os.makedirs('Emission_DNN', exist_ok=True)
#     if input("Enter 'C' to continue, else any: ").lower()=='c':
#         ck = 'Emission_DNN/dnn_state'
#         if os.path.exists(ck):
#             model.load_state_dict(torch.load(ck, map_location=device))

#     # 10) Train!
#     (train_losses, train_accs, train_errs,
#      valid_losses, valid_accs, valid_errs) = train_model(
#         model,
#         train_loader=train_loader,
#         valid_loader=valid_loader,
#         learning_rate=1e-5,
#         epochs=150,
#         weight_decay=1e-4,
#         checkpoint_path='Emission_DNN/dnn_state'
#     )

#     print(f"Final Train ¡ú Loss: {train_losses[-1]:.4f}, "
#           f"Within-1: {train_accs[-1]*100:.2f}%, "
#           f"MeanErr: {train_errs[-1]:.2f}px")
#     print(f"Final Valid ¡ú Loss: {valid_losses[-1]:.4f}, "
#           f"Within-1: {valid_accs[-1]*100:.2f}%, "
#           f"MeanErr: {valid_errs[-1]:.2f}px")


# Train_Model.py V3
# This script trains a UNet emission model for track-before-detect.

# import os
# import torch
# import numpy as np
# from torch.utils.data import TensorDataset, DataLoader
# from unet_model import UNet
# from train_parts import train_model


# def to_numeric_array(x, dtype=np.float32):
#     """
#     Convert nested sequences or object-dtype arrays to a numeric numpy array.
#     Supports lists, tuples, or object arrays by stacking individual float32 arrays.
#     """
#     if isinstance(x, np.ndarray) and x.dtype != object:
#         return x.astype(dtype)
#     seq = None
#     if isinstance(x, (list, tuple)) or (isinstance(x, np.ndarray) and x.dtype == object):
#         seq = list(x)
#     if seq is not None:
#         arrs = [np.array(elem, dtype=dtype) for elem in seq]
#         return np.stack(arrs, axis=0)
#     raise ValueError("Input cannot be converted to a numeric array")


# def make_gaussian_heatmap_array(coords, H, W, sigma=3.0):
#     """
#     Generate a batch of 2D Gaussian heatmaps for each coordinate in coords.

#     Args:
#         coords: Array of shape (N, 2) with (col, row) pixel indices.
#         H: Height of each heatmap.
#         W: Width of each heatmap.
#         sigma: Standard deviation of the Gaussian.

#     Returns:
#         heatmaps: numpy array of shape (N, 1, H, W).
#     """
#     N = coords.shape[0]
#     ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
#     maps = np.zeros((N, 1, H, W), dtype=np.float32)
#     for i, (c, r) in enumerate(coords):
#         g = np.exp(-((xs - r)**2 + (ys - c)**2) / (2 * sigma**2))
#         m = g.max()
#         maps[i, 0] = g / m if m > 0 else g
#     return maps


# if __name__ == "__main__":
#     # Determine device (GPU if available)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 1) Load raw data from preprocessed .pt files
#     train_data_raw  = torch.load('train_data.pt',   weights_only=False)
#     valid_data_raw  = torch.load('valid_data.pt',   weights_only=False)
#     train_label_raw = torch.load('train_label.pt',  weights_only=False)
#     valid_label_raw = torch.load('valid_label.pt',  weights_only=False)

#     # 2) Extract last-frame physical coordinates and convert data to numpy
#     train_label_phys = to_numeric_array(train_label_raw, dtype=np.float32)[:, -1, :]
#     valid_label_phys = to_numeric_array(valid_label_raw, dtype=np.float32)[:, -1, :]
#     train_data_np    = to_numeric_array(train_data_raw,  dtype=np.float32)
#     valid_data_np    = to_numeric_array(valid_data_raw,  dtype=np.float32)

#     # 3) Discretize physical positions into pixel indices
#     N, K, H, W = train_data_np.shape  # batch size, window length, height, width
#     R_min, R_max = 1000.0, 2500.0     # range limits in meters
#     V_max        = 300.0              # max velocity in m/s
#     dr = (R_max - R_min) / (W - 1)
#     dv = (2 * V_max)   / (H - 1)

#     r_idx_train = np.clip(np.round((train_label_phys[:, 0] - R_min) / dr), 0, W-1).astype(int)
#     v_idx_train = np.clip(np.round((train_label_phys[:, 1] + V_max)   / dv), 0, H-1).astype(int)
#     r_idx_valid = np.clip(np.round((valid_label_phys[:, 0] - R_min) / dr), 0, W-1).astype(int)
#     v_idx_valid = np.clip(np.round((valid_label_phys[:, 1] + V_max)   / dv), 0, H-1).astype(int)

#     train_coords = np.stack([r_idx_train, v_idx_train], axis=1)
#     valid_coords = np.stack([r_idx_valid, v_idx_valid], axis=1)

#     # 4) Build Gaussian heatmaps for supervision
#     train_heatmap_np = make_gaussian_heatmap_array(train_coords, H, W, sigma=3.0)
#     valid_heatmap_np = make_gaussian_heatmap_array(valid_coords, H, W, sigma=3.0)

#     # 5) Convert data and heatmaps to torch tensors and move to device
#     train_data_t    = torch.from_numpy(train_data_np).to(device)
#     train_heatmap_t = torch.from_numpy(train_heatmap_np).to(device)
#     valid_data_t    = torch.from_numpy(valid_data_np).to(device)
#     valid_heatmap_t = torch.from_numpy(valid_heatmap_np).to(device)

#     # 6) Normalize each sample independently (zero mean, unit variance)
#     for t in (train_data_t, valid_data_t):
#         mean = t.mean(dim=[2,3], keepdim=True)
#         std  = t.std(dim=[2,3], keepdim=True) + 1e-6
#         t.sub_(mean).div_(std)

#     # 7) Create DataLoaders for training and validation
#     batch_size = 32
#     train_loader = DataLoader(TensorDataset(train_data_t, train_heatmap_t),
#                               batch_size=batch_size, shuffle=True)
#     valid_loader = DataLoader(TensorDataset(valid_data_t, valid_heatmap_t),
#                               batch_size=len(valid_data_t), shuffle=False)

#     # 8) Initialize UNet with dynamic window length and feature map size
#     #    in_channels=K (number of frames), H and W are number of bins in each dimension
#     model = UNet(in_channels=K, base_channels=64, bilinear=True, H=H, W=W)
#     model.to(device)

#     # 9) Optionally resume from checkpoint if exists
#     os.makedirs('Emission_DNN', exist_ok=True)
#     if input("Enter 'C' to continue from checkpoint, else press Enter: ").lower() == 'c':
#         ckpt_path = 'Emission_DNN/dnn_state'
#         if os.path.exists(ckpt_path):
#             print("Loading checkpoint...")
#             model.load_state_dict(torch.load(ckpt_path, map_location=device))

#     # 10) Train the model and save checkpoints
#     train_losses, train_accs, train_errs, \
#     valid_losses, valid_accs, valid_errs = train_model(
#         model,
#         train_loader=train_loader,
#         valid_loader=valid_loader,
#         learning_rate=1e-5,
#         epochs=150,
#         weight_decay=1e-4,
#         checkpoint_path='Emission_DNN/dnn_state'
#     )

#     # Print final metrics
#     print(f"Final Training Loss: {train_losses[-1]:.4f}, "
#           f"Accuracy (<1 bin): {train_accs[-1]*100:.2f}%, "
#           f"Mean Error: {train_errs[-1]:.2f} bins")
#     print(f"Final Validation Loss: {valid_losses[-1]:.4f}, "
#           f"Accuracy (<1 bin): {valid_accs[-1]*100:.2f}%, "
#           f"Mean Error: {valid_errs[-1]:.2f} bins")


# # Train_Model.py V4
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from unet_model import UNet
from train_parts import train_model

def to_numeric_array(x, dtype=np.float32):
    """
    Convert nested sequences or object-dtype arrays to a numeric numpy array.
    Supports lists, tuples, or object arrays by stacking individual float32 arrays.
    """
    if isinstance(x, np.ndarray) and x.dtype != object:
        return x.astype(dtype)
    seq = None
    if isinstance(x, (list, tuple)) or (isinstance(x, np.ndarray) and x.dtype == object):
        seq = list(x)
    if seq is not None:
        arrs = [np.array(elem, dtype=dtype) for elem in seq]
        return np.stack(arrs, axis=0)
    raise ValueError("Input cannot be converted to a numeric array")

def make_gaussian_heatmap_array(coords, H, W, sigma=3.0):
    """
    Generate a batch of 2D Gaussian heatmaps for each coordinate in coords.

    Args:
        coords: Array of shape (N, 2) with (col, row) pixel indices.
        H: Height of each heatmap.
        W: Width of each heatmap.
        sigma: Standard deviation of the Gaussian.

    Returns:
        heatmaps: numpy array of shape (N, 1, H, W).
    """
    N = coords.shape[0]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    maps = np.zeros((N, 1, H, W), dtype=np.float32)
    for i, (c, r) in enumerate(coords):
        g = np.exp(-((xs - r)**2 + (ys - c)**2) / (2 * sigma**2))
        m = g.max()
        maps[i, 0] = g / m if m > 0 else g
    return maps

if __name__ == "__main__":
    # Determine device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load raw data from preprocessed .pt files
    train_data_raw  = torch.load('train_data.pt',   weights_only=False)
    valid_data_raw  = torch.load('valid_data.pt',   weights_only=False)
    train_label_raw = torch.load('train_label.pt',  weights_only=False)
    valid_label_raw = torch.load('valid_label.pt',  weights_only=False)

    # 2) Extract last-frame physical coordinates and convert data to numpy
    train_label_phys = to_numeric_array(train_label_raw, dtype=np.float32)[:, -1, :]
    valid_label_phys = to_numeric_array(valid_label_raw, dtype=np.float32)[:, -1, :]
    train_data_np    = to_numeric_array(train_data_raw,  dtype=np.float32)
    valid_data_np    = to_numeric_array(valid_data_raw,  dtype=np.float32)

    # 3) Discretize physical positions into pixel indices
    N, K, H, W = train_data_np.shape  # batch size, window length, height, width
    R_min, R_max = 1000.0, 2500.0     # range limits in meters
    V_max        = 300.0              # max velocity in m/s
    dr = (R_max - R_min) / (W - 1)
    dv = (2 * V_max)   / (H - 1)

    r_idx_train = np.clip(np.round((train_label_phys[:, 0] - R_min) / dr), 0, W-1).astype(int)
    v_idx_train = np.clip(np.round((train_label_phys[:, 1] + V_max)   / dv), 0, H-1).astype(int)
    r_idx_valid = np.clip(np.round((valid_label_phys[:, 0] - R_min) / dr), 0, W-1).astype(int)
    v_idx_valid = np.clip(np.round((valid_label_phys[:, 1] + V_max)   / dv), 0, H-1).astype(int)

    train_coords = np.stack([r_idx_train, v_idx_train], axis=1)
    valid_coords = np.stack([r_idx_valid, v_idx_valid], axis=1)

    # 4) Build Gaussian heatmaps for supervision
    train_heatmap_np = make_gaussian_heatmap_array(train_coords, H, W, sigma=1.0)
    valid_heatmap_np = make_gaussian_heatmap_array(valid_coords, H, W, sigma=1.0)

    # 5) Convert data and heatmaps to torch tensors and move to device
    train_data_t    = torch.from_numpy(train_data_np).to(device)
    train_heatmap_t = torch.from_numpy(train_heatmap_np).to(device)
    valid_data_t    = torch.from_numpy(valid_data_np).to(device)
    valid_heatmap_t = torch.from_numpy(valid_heatmap_np).to(device)

    # 6) Normalize each sample independently (zero mean, unit variance)
    for t in (train_data_t, valid_data_t):
        mean = t.mean(dim=[2,3], keepdim=True)
        std  = t.std(dim=[2,3], keepdim=True) + 1e-6
        t.sub_(mean).div_(std)

    # 7) Create DataLoaders for training and validation
    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_data_t, train_heatmap_t),
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_data_t, valid_heatmap_t),
                              batch_size=len(valid_data_t), shuffle=False)

    # 8) Initialize UNet with dynamic window length and feature map size
    print(f">>> UNet config: in_channels={K}, H={H}, W={W}")
    model = UNet(in_channels=K, base_channels=64, bilinear=True, H=H, W=W)
    model.to(device)

    # 9) Optionally resume from checkpoint if exists
    os.makedirs('Emission_DNN', exist_ok=True)
    ckpt_path = 'Emission_DNN/dnn_state.pth'
    if input("Enter 'C' to continue from checkpoint, else press Enter: ").lower() == 'c':
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # 10) Train the model and save checkpoints
    train_losses, train_accs, train_errs, \
    valid_losses, valid_accs, valid_errs = train_model(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        learning_rate=1e-5,
        epochs=1000,
        weight_decay=1e-4,
        checkpoint_path=ckpt_path
    )

    # Print final metrics
    print(f"Final Training Loss: {train_losses[-1]:.4f}, "
          f"Accuracy (<1 bin): {train_accs[-1]*100:.2f}%, "
          f"Mean Error: {train_errs[-1]:.2f} bins")
    print(f"Final Validation Loss: {valid_losses[-1]:.4f}, "
          f"Accuracy (<1 bin): {valid_accs[-1]*100:.2f}%, "
          f"Mean Error: {valid_errs[-1]:.2f} bins")
