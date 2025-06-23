# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from unet_model import UNet
# from Tracker_parts import TransDist, Tracker
# import matplotlib.pyplot as plt



# data_file = '/home/zktx/YSX/YSX_TBD/test_data_phys.pt'
# data_dict = torch.load(data_file, weights_only=False)
# all_data = data_dict['data']     # list, len=4750, each [5, 64, 64]
# all_labels = data_dict['labels'] # list, len=4750, each [2]

# print("Number of samples:", len(all_data))
# print("Number of frames per sample:", len(all_data[0]))
# print("Shape of each frame:", np.array(all_data[0][0]).shape)


# window_size = 5
# Nr = 200
# Nv = 64
# model_path = '/home/zktx/YSX/YSX_TBD/Emission_DNN/dnn_state'
# state_dict = torch.load(model_path)

# def remove_module_prefix(state_dict):
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         new_key = k[7:] if k.startswith('module.') else k
#         new_state_dict[new_key] = v
#     return new_state_dict

# state_dict = remove_module_prefix(state_dict)
# emis_DNN = UNet(in_channels=window_size, n=64, Nr=Nr, Nv=Nv)
# emis_DNN.load_state_dict(state_dict, strict=False)
# emis_DNN.eval()


# batch_size = 1
# loader = DataLoader(list(zip(all_data, all_labels)), batch_size=batch_size, shuffle=False)


# transdist = TransDist(Nr=Nr, Nv=Nv, sigma_r=30, sigma_v=20, T=0)


# all_rmse_range = []
# all_rmse_vel = []

# for i, (input, label) in enumerate(loader):
#     cheat_state = label[0]
#     input = torch.tensor(np.array(input), dtype=torch.float32)   # [1, 5, 64, 64]
    
    
    
#     with torch.no_grad():
#         output = emis_DNN(input)  # [1, 2, Nr, Nv]
#         emis_seq = output[0].unsqueeze(0)  # [1, 2, Nr, Nv] (batch=1, window=1)
#         viterbi = Tracker(transdist)
#         estim_vit = viterbi.Viterbi(emis_seq, cheat_state=cheat_state, bbox_drop=True)
      
#     label_tuples = [(int(label[0][0]), int(label[0][1]))] * window_size
#     range_vec = transdist.range_vec.cpu().numpy()
#     vr_vec    = transdist.vr_vec.cpu().numpy()
    
#     r_pred_vals = [range_vec[rp] for (rp,_) in estim_vit]
#     v_pred_vals = [   vr_vec[vp] for (_,vp) in estim_vit]

#     r_true_val = label[0][0].item()
#     v_true_val = label[0][1].item()

#     rmse_r = np.sqrt(np.mean([(rp - r_true_val)**2 for rp in r_pred_vals]))
#     rmse_v = np.sqrt(np.mean([(vp - v_true_val)**2 for vp in v_pred_vals]))

#     all_rmse_range.append(rmse_r)
#     all_rmse_vel.append(rmse_v)
    
#     if i < len(loader):
#         print(f"Sample {i}: RMSE_range={rmse_r:.2f}, RMSE_vel={rmse_v:.2f}, gt={label_tuples[0]}, pred={estim_vit[0]}")
    
# print(f"RMSE-Range: {np.mean(all_rmse_range):.2f} m")
# print(f"RMSE-Vel: {np.mean(all_rmse_vel):.2f} m/s")
# print("All done!")

# plt.figure()
# plt.plot(all_rmse_range, marker='o')
# plt.xlabel('Sample Index')
# plt.ylabel('RMSE Range (m)')
# plt.title('Tracking RMSE in Range over All Samples')
# plt.grid(True)
# plt.show()


# plt.figure()
# plt.plot(all_rmse_vel, marker='o')
# plt.xlabel('Sample Index')
# plt.ylabel('RMSE Velocity (m/s)')
# plt.title('Tracking RMSE in Velocity over All Samples')
# plt.grid(True)
# plt.show()





# experiment_viterbi.py V2
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from unet_model import UNet
# from Tracker_parts import TransDist

# class Tracker:
#     def __init__(self, transdist):
#         self.trans = transdist

#     def Viterbi(self, emis_seq, cheat_state=None, bbox_drop=False):
#         """
#         Run simple per©\time MAP (argmax) over the emission sequence.
#         emis_seq: shape [1, T, H, W]
#         Returns list of (r_bin, v_bin) of length T.
#         """
#         seq_len = emis_seq.shape[1]
#         est = []
#         for t in range(seq_len):
#             emis = emis_seq[0, t]       # [H, W]
#             flat = int(emis.argmax())
#             H, W = emis.shape
#             r = flat // W
#             v = flat % W
#             est.append((r, v))
#         return est

# def remove_module_prefix(state_dict):
#     new_state = {}
#     for k, v in state_dict.items():
#         new_key = k[7:] if k.startswith('module.') else k
#         new_state[new_key] = v
#     return new_state

# if __name__ == "__main__":
#     # 1) Config & Paths
#     device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     data_file   = '/home/zktx/YSX/YSX_TBD/test_data_phys.pt'
#     model_path  = '/home/zktx/YSX/YSX_TBD/Emission_DNN/dnn_state.pth'
#     window_size = 5       # sliding©\window length
#     Nr, Nv      = 200, 64 # grid dims
#     sigma_r     = 30
#     sigma_v     = 20
#     T_dummy     = 0

#     # 2) Load Test Data
#     data_dict   = torch.load(data_file, weights_only=False)
#     all_data    = data_dict['data']    # N sequences, each a list of K frames
#     all_labels  = data_dict['labels']  # N labels, each [r_true, v_true]
#     print(f"Loaded {len(all_data)} sequences, each with {window_size} frames, grid {Nr}¡Á{Nv}.")

#     # 3) Build & Load UNet
#     emis_DNN = UNet(
#         in_channels=window_size,
#         base_channels=64,
#         bilinear=True,
#         H=Nr,
#         W=Nv
#     ).to(device)
#     print(f">>> UNet config matched: in_channels={window_size}, H={Nr}, W={Nv}")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"No checkpoint found at {model_path}")
#     raw_state  = torch.load(model_path, map_location=device)
#     state      = remove_module_prefix(raw_state)
#     model_dict = emis_DNN.state_dict()
#     # Only load layers whose shapes match
#     filtered = {
#         k: v for k, v in state.items()
#         if k in model_dict and v.size() == model_dict[k].size()
#     }
#     model_dict.update(filtered)
#     emis_DNN.load_state_dict(model_dict)
#     emis_DNN.eval()
#     print(f"Loaded {len(filtered)}/{len(model_dict)} matching layers from checkpoint.")

#     # 4) Prepare Tracker
#     transdist = TransDist(Nr=Nr, Nv=Nv, sigma_r=sigma_r, sigma_v=sigma_v, T=T_dummy)
#     tracker   = Tracker(transdist)

#     # 5) Inference with sliding©\window + full©\sequence Viterbi
#     all_rmse_range = []
#     all_rmse_vel   = []
#     K_win = window_size

#     for idx, (frames, label) in enumerate(zip(all_data, all_labels)):
#         # frames: list of K_win arrays shape (H, W)
#         T_frames     = len(frames)
#         emission_seq = []

#         # build emission map for each sliding window
#         for t in range(K_win, T_frames + 1):
#             win = np.stack(frames[t - K_win:t], axis=0)            # (K_win, H, W)
#             x   = torch.from_numpy(win).float().unsqueeze(0).to(device)  # [1, K_win, H, W]

#             # normalize (same as training)
#             m = x.mean(dim=[2, 3], keepdim=True)
#             s = x.std(dim=[2, 3], keepdim=True) + 1e-6
#             x_norm = (x - m) / s

#             with torch.no_grad():
#                 logits = emis_DNN(x_norm)                         # [1, H*W]
#                 heat   = torch.sigmoid(logits).view(1, 1, Nr, Nv) # [1,1,H,W]
#             emission_seq.append(heat.cpu().numpy()[0, 0])       # [H, W]

#         emission_seq = np.stack(emission_seq, axis=0)  # [T_frames-K_win+1, H, W]

#         # full©\sequence Viterbi
#         path = tracker.Viterbi(emission_seq[np.newaxis, ...])  # list length T_frames-K_win+1

#         # pick the *last* time©\step prediction
#         r_bin, v_bin   = path[-1]
#         range_vec      = transdist.range_vec.cpu().numpy()
#         vr_vec         = transdist.vr_vec.cpu().numpy()
#         r_pred, v_pred = range_vec[r_bin], vr_vec[v_bin]

#         # CORRECT UNPACK of the single [r_true, v_true] label
#         r_true, v_true = label

#         # absolute©\error
#         err_r = abs(r_pred - r_true)
#         err_v = abs(v_pred - v_true)
#         all_rmse_range.append(err_r)
#         all_rmse_vel.append(err_v)

#         print(f"[{idx:04d}] RMSE_range={err_r:.2f} m, RMSE_vel={err_v:.2f} m/s, "
#               f"gt=({r_true:.1f},{v_true:.1f}), pred=({r_pred:.1f},{v_pred:.1f})")

#     # 6) Summary & Plots
#     print("\n=== Final Results ===")
#     print(f"Mean abs error Range: {np.mean(all_rmse_range):.2f} m")
#     print(f"Mean abs error Vel:   {np.mean(all_rmse_vel):.2f} m/s")

#     plt.figure(figsize=(8,4))
#     plt.plot(all_rmse_range, marker='o')
#     plt.xlabel('Sequence Index')
#     plt.ylabel('Absolute Range Error (m)')
#     plt.title('Sliding©\Window UNet Range Error')
#     plt.grid(True)

#     plt.figure(figsize=(8,4))
#     plt.plot(all_rmse_vel, marker='o')
#     plt.xlabel('Sequence Index')
#     plt.ylabel('Absolute Velocity Error (m/s)')
#     plt.title('Sliding©\Window UNet Velocity Error')
#     plt.grid(True)

#     plt.show()



# experiment_viterbi.py V3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet
from Tracker_parts import TransDist, Track

class Tracker:
    def __init__(self, transdist):
        P       = transdist.get_transition_matrix()  # (S,S)
        self.logA = np.log(P + 1e-12).astype(np.float32)

    def Viterbi(self, log_emis_seq):
        T, H, W = log_emis_seq.shape
        S       = H * W
        logE    = log_emis_seq.reshape(T, S)

        dp      = np.full((T, S), -np.inf, dtype=np.float32)
        backptr = np.zeros((T, S),   dtype=np.int32)

        dp[0] = logE[0]
        for t in range(1, T):
            scores     = dp[t-1][:, None] + self.logA
            backptr[t] = np.argmax(scores, axis=0)
            dp[t]      = scores[backptr[t], np.arange(S)] + logE[t]

        path_states = np.zeros(T, dtype=np.int32)
        path_states[-1] = np.argmax(dp[-1])
        for t in range(T-1, 0, -1):
            path_states[t-1] = backptr[t, path_states[t]]

        return [(s//W, s%W) for s in path_states]


def remove_module_prefix(state_dict):
    new_state = {}
    for k,v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state[new_key] = v
    return new_state


if __name__ == "__main__":
    # 1) Config
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_file  = '/home/zktx/YSX/YSX_TBD/test_data_phys.pt'
    model_path = '/home/zktx/YSX/YSX_TBD/Emission_DNN/dnn_state.pth'
    Nr, Nv    = 200, 64
    sigma_r   = 30
    sigma_v   = 20
    N_chirps  = 64
    PRI       = 20e-6
    T         = N_chirps * PRI

    # 2) Load data
    data_dict  = torch.load(data_file, weights_only=False)
    all_data   = data_dict['data']
    all_labels = data_dict['labels']
    window_size= len(all_data[0])
    K_win      = window_size
    print(f"Loaded {len(all_data)} samples, {window_size} frames each, grid {Nr}¡Á{Nv}.")

    # 3) Build & load model
    emis_DNN = UNet(
        in_channels = window_size,
        base_channels=128,
        bilinear    = True,
        H           = Nr,
        W           = Nv
    ).to(device)
    raw_state = torch.load(model_path, map_location=device)
    state     = remove_module_prefix(raw_state)
    mdict     = emis_DNN.state_dict()
    filtered  = {k:v for k,v in state.items() if k in mdict and v.size()==mdict[k].size()}
    mdict.update(filtered)
    emis_DNN.load_state_dict(mdict)
    emis_DNN.eval()
    print(f"Loaded {len(filtered)}/{len(mdict)} layers.")

    # 4) Prepare tracker
    transdist = TransDist(Nr=Nr, Nv=Nv, sigma_r=sigma_r, sigma_v=sigma_v, T=T)
    tracker   = Tracker(transdist)

    # 5) Inference
    all_rmse_r = []
    all_rmse_v = []

    for idx, (frames, label) in enumerate(zip(all_data, all_labels)):
        emission_seq = []
        for t in range(K_win, len(frames)+1):
            win = np.stack(frames[t-K_win:t], axis=0)
            x   = torch.from_numpy(win).float().unsqueeze(0).to(device)
            m = x.mean(dim=[2,3], keepdim=True)
            s = x.std(dim=[2,3], keepdim=True) + 1e-6
            x = (x - m) / s

            with torch.no_grad():
                rlogits, vlogits = emis_DNN(x)  # (1,H),(1,W)
                pr = torch.sigmoid(rlogits).cpu().numpy().ravel()
                pv = torch.sigmoid(vlogits).cpu().numpy().ravel()
                emis = np.outer(pr, pv)         # joint emission (H,W)
            emission_seq.append(emis)

        prob_seq = np.stack(emission_seq, axis=0)     # (T_eff,H,W)
        log_emis = np.log(prob_seq + 1e-12).astype(np.float32)
        path     = tracker.Viterbi(log_emis)

        # take last step
        rbin, vbin = path[-1]
        rvals = transdist.range_vec.cpu().numpy()
        vvals = transdist.vr_vec.cpu().numpy()
        r_pred, v_pred = rvals[rbin], vvals[vbin]
        r_true, v_true = label

        all_rmse_r.append(abs(r_pred - r_true))
        all_rmse_v.append(abs(v_pred - v_true))

        print(f"[{idx:04d}] RMSE_r={all_rmse_r[-1]:.2f} m, "
              f"RMSE_v={all_rmse_v[-1]:.2f} m/s, "
              f"gt=({r_true:.1f},{v_true:.1f}), "
              f"pred=({r_pred:.1f},{v_pred:.1f})")

    # 6) Summary
    print("\n=== Final Results ===")
    print(f"Mean abs error Range: {np.mean(all_rmse_r):.2f} m")
    print(f"Mean abs error Vel:   {np.mean(all_rmse_v):.2f} m/s")

    # plot
    plt.figure(figsize=(8,4))
    plt.plot(all_rmse_r, marker='o')
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Range Error (m)')
    plt.title('UNet Emission Absolute Range Error')
    plt.grid(True)

    plt.figure(figsize=(8,4))
    plt.plot(all_rmse_v, marker='o')
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Velocity Error (m/s)')
    plt.title('UNet Emission Absolute Velocity Error')
    plt.grid(True)

    plt.show()
