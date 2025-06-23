import h5py 
import numpy as np
import torch

file_name = '/home/zktx/YSX/YSX_TBD/simulate_data/test_data_phys.mat'  # your test .mat file name

with h5py.File(file_name, 'r') as f:
    print("Keys in file:", list(f.keys()))
    data = f['data']
    labels = f['labels']
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    n_samples, n_frames = data.shape
    example_frame = np.array(f[data[0, 0]])
    frame_shape = example_frame.shape
    print("Frame shape:", frame_shape)

    all_data = []
    all_labels = []

    for i in range(n_samples):
        traj_data = []
        traj_labels = []
        
        # For each sample, gather the frames for the input and the corresponding labels
        for k in range(n_frames):
            frame = np.array(f[data[i, k]])  # shape: (64, 64)
            traj_data.append(frame)
            label = np.array(f[labels[i, k]]).squeeze()  # label is a scalar
            traj_labels.append(label)

        # Now stack 5 frames into 1 input (5-channel image) for each trajectory
        for j in range(n_frames - 5):  # We take sequences of 5 consecutive frames
            frames_5 = np.stack(traj_data[j:j+5], axis=0)  # Stack 5 consecutive frames
            all_data.append(frames_5)  # Shape: [5, 64, 64]
            all_labels.append(traj_labels[j+5-1])  # Label for the 5th frame
        
        if i % 10 == 0:
            print(f"Loaded {i}/{n_samples} trajectories")

# Ensure that the input size is large enough for convolution
# Convert data into 4D tensor format [N, C, H, W] and resize to (64, 64) to ensure it's large enough
all_data_resized = []
for frames in all_data:
    frames_resized = []
    for frame in frames:
        # Add a batch and channel dimension: [1, 1, 64, 64]
        frame = torch.tensor(frame).unsqueeze(0).unsqueeze(0)
        
        # Resize the frame to (64, 64) to ensure it's large enough for convolution
        frame_resized = torch.nn.functional.interpolate(frame, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Remove batch and channel dimensions to return to [64, 64]
        frames_resized.append(frame_resized.squeeze(0).squeeze(0))  # [64, 64]
    
    all_data_resized.append(np.stack(frames_resized, axis=0))  # Shape: [5, 64, 64]

# Save the processed data
torch.save({'data': all_data_resized, 'labels': all_labels}, 'test_data_phys.pt')
print("All test data has been loaded and saved as test_data_phys.pt!")