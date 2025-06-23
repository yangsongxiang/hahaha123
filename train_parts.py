#train_part.py V1
# import torch
# import time
# import torch.nn.functional as F
# import datetime
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import math

# def train_model(model,
#                 train_loader: DataLoader,
#                 valid_loader: DataLoader,
#                 learning_rate=0.001,
#                 epochs=3000,
#                 weight_decay: float = 1,
#                 checkpoint_path=None):

#     train_loss_list, train_acc_list = [], []
#     valid_loss_list, valid_acc_list = [], []

#     optimizer = optim.Adam(model.parameters(),
#                            lr=learning_rate,
#                            weight_decay=weight_decay,
#                            betas=(0.90, 0.999))

#     # We¡¯ll need W (width) to reconstruct flat label indices:
#     # Since your UNet stores H,W as attributes:
#     H, W = model.H, model.W
#     HW = H * W

#     for epoch in range(epochs):
#         start_time = time.time()
#         train_loss = 0.0
#         train_acc  = 0.0

#         # --- TRAINING ---
#         model.train()
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)  # shape: (batch, H*W)

#             # build flat label index: d*W + r
#             flat_labels = (labels[:,1] * W + labels[:,0]).to(outputs.device)

#             # single-line cross-entropy
#             loss = F.cross_entropy(outputs, flat_labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             # accuracy as fraction of exact matches
#             preds = outputs.argmax(dim=1)
#             train_acc += (preds == flat_labels).float().mean().item()

#         # --- VALIDATION ---
#         valid_loss = 0.0
#         valid_acc  = 0.0
#         model.eval()
#         with torch.no_grad():
#             for inputs, labels in valid_loader:
#                 outputs = model(inputs)
#                 flat_labels = (labels[:,1] * W + labels[:,0]).to(outputs.device)

#                 loss = F.cross_entropy(outputs, flat_labels)
#                 valid_loss += loss.item()

#                 preds = outputs.argmax(dim=1)
#                 valid_acc += (preds == flat_labels).float().mean().item()

#         # average over batches
#         train_loss /= len(train_loader)
#         train_acc  /= len(train_loader)
#         valid_loss /= len(valid_loader)
#         valid_acc  /= len(valid_loader)

#         train_loss_list.append(train_loss)
#         train_acc_list.append(train_acc)
#         valid_loss_list.append(valid_loss)
#         valid_acc_list.append(valid_acc)

#         elapsed   = time.time() - start_time
#         remaining = elapsed * (epochs - epoch - 1)
#         print(f"Epoch {epoch+1}/{epochs}  "
#               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%  "
#               f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc*100:.2f}%  "
#               f"Time: {format_time(elapsed)}, ETA: {format_time(remaining)}")

#         if checkpoint_path:
#             torch.save(model.state_dict(), checkpoint_path)

#     return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


# def format_time(seconds):
#     return str(datetime.timedelta(seconds=int(seconds)))

# train_parts.py V2
import time
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim

def train_model(
    model: torch.nn.Module,
    train_loader,
    valid_loader,
    learning_rate: float = 1e-4,    # raised LR
    epochs: int = 300,               # train longer
    weight_decay: float = 1e-5,      # reduced weight decay
    checkpoint_path: str = None
):
    """
    Train UNet with two-head CE loss: range + velocity.
    Reports within-1-bin accuracy and mean-pixel error.
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    H, W = model.H, model.W

    train_losses, train_accs, train_errs = [], [], []
    valid_losses, valid_accs, valid_errs = [], [], []

    for epoch in range(epochs):
        t0 = time.time()
        running_loss = 0.0
        running_acc  = 0.0
        running_err  = 0.0

        # --- TRAIN ---
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # two-head forward
            rlogits, vlogits = model(inputs)  # shapes (B,H), (B,W)

            # build flat bin index from one-hot heatmap
            B = targets.size(0)
            flat = targets.view(B, H*W).argmax(dim=1)   # (B,)
            r_tgt = (flat // W).to(rlogits.device)      # (B,)
            v_tgt = (flat %  W).to(vlogits.device)      # (B,)

            # CE losses
            loss_r = F.cross_entropy(rlogits, r_tgt)
            loss_v = F.cross_entropy(vlogits, v_tgt)
            loss   = loss_r + loss_v

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            # compute within-1-bin accuracy & mean error
            rpred = rlogits.argmax(dim=1)
            vpred = vlogits.argmax(dim=1)
            dr = (rpred.float() - r_tgt.float()).abs()
            dv = (vpred.float() - v_tgt.float()).abs()
            dist = torch.sqrt(dr**2 + dv**2)
            running_acc += (dist <= 1.0).float().mean().item()
            running_err += dist.mean().item()

        # --- VALID ---
        val_loss, val_acc, val_err = 0.0, 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                rlogits, vlogits = model(inputs)
                B = targets.size(0)
                flat = targets.view(B, H*W).argmax(dim=1)
                r_tgt = (flat // W).to(rlogits.device)
                v_tgt = (flat %  W).to(vlogits.device)

                loss_r = F.cross_entropy(rlogits, r_tgt)
                loss_v = F.cross_entropy(vlogits, v_tgt)
                loss   = loss_r + loss_v
                val_loss += loss.item()

                rpred = rlogits.argmax(dim=1)
                vpred = vlogits.argmax(dim=1)
                dr = (rpred.float() - r_tgt.float()).abs()
                dv = (vpred.float() - v_tgt.float()).abs()
                dist = torch.sqrt(dr**2 + dv**2)
                val_acc += (dist <= 1.0).float().mean().item()
                val_err += dist.mean().item()

        # average and log
        train_losses.append(running_loss / len(train_loader))
        train_accs  .append(running_acc  / len(train_loader))
        train_errs  .append(running_err  / len(train_loader))
        valid_losses.append(val_loss   / len(valid_loader))
        valid_accs  .append(val_acc     / len(valid_loader))
        valid_errs  .append(val_err     / len(valid_loader))

        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"Train Loss: {train_losses[-1]:.4f}, "
            f"Within-1: {train_accs[-1]*100:.2f}%, "
            f"MeanErr: {train_errs[-1]:.2f}px  |  "
            f"Valid Loss: {valid_losses[-1]:.4f}, "
            f"Within-1: {valid_accs[-1]*100:.2f}%, "
            f"MeanErr: {valid_errs[-1]:.2f}px  |  "
            f"Time: {str(datetime.timedelta(seconds=int(dt)))}"
        )

        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)

    return (
        train_losses, train_accs, train_errs,
        valid_losses, valid_accs, valid_errs
    )

