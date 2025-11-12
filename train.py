import os
import argparse
import time
import copy
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from lifelines.utils import concordance_index

# Import your model
from models.unified_model import UnifiedMultimodalModel
from models.survival_head import cox_partial_log_likelihood


# -----------------------------
# Dataset
# -----------------------------
class MultimodalNPZDataset(Dataset):
    """
    Loads data saved as a .npz file with arrays:
      mod_0, mod_1, ..., labels, durations, events
    """

    def __init__(self, npz_path: str, modality_keys: List[str] = None):
        data = np.load(npz_path, allow_pickle=True)
        # If modality_keys provided, use them; else auto-detect mod_ prefix
        if modality_keys:
            self.modalities = [data[k].astype(np.float32) for k in modality_keys]
        else:
            # detect all keys starting with 'mod_'
            mod_keys = sorted([k for k in data.keys() if k.startswith("mod_")])
            if len(mod_keys) == 0:
                raise ValueError(f"No modality arrays found in {npz_path}. Expected keys mod_0, mod_1, ...")
            self.modalities = [data[k].astype(np.float32) for k in mod_keys]

        # Required arrays
        if "labels" not in data or "durations" not in data or "events" not in data:
            raise ValueError(f"{npz_path} must contain 'labels', 'durations' and 'events' arrays.")
        self.labels = data["labels"].astype(np.int64)
        self.durations = data["durations"].astype(np.float32)
        self.events = data["events"].astype(np.float32)

        # number of samples
        self.N = self.labels.shape[0]
        # sanity checks: all modality arrays must have same N
        for arr in self.modalities:
            if arr.shape[0] != self.N:
                raise ValueError("All modality arrays must have same number of samples as labels.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # return list of modality arrays for index idx, plus label, duration, event
        mods = [torch.from_numpy(arr[idx]) for arr in self.modalities]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        duration = torch.tensor(self.durations[idx], dtype=torch.float32)
        event = torch.tensor(self.events[idx], dtype=torch.float32)
        return mods, label, duration, event


# -----------------------------
# Collate fn for DataLoader
# -----------------------------
def multimodal_collate(batch):
    """
    Batch is list of tuples (mods_list, label, duration, event)
    We need to stack each modality separately.
    """
    batch_size = len(batch)
    num_modalities = len(batch[0][0])
    # prepare lists for each modality
    mods_per_modality = [[] for _ in range(num_modalities)]
    labels = []
    durations = []
    events = []

    for mods, lbl, dur, ev in batch:
        for i, m in enumerate(mods):
            mods_per_modality[i].append(m)
        labels.append(lbl)
        durations.append(dur)
        events.append(ev)

    # stack modalities
    mods_stacked = [torch.stack(lst, dim=0) for lst in mods_per_modality]
    labels = torch.stack(labels, dim=0)
    durations = torch.stack(durations, dim=0)
    events = torch.stack(events, dim=0)

    return mods_stacked, labels, durations, events


# -----------------------------
# Utility: compute metrics
# -----------------------------
def compute_classification_metrics(logits, labels):
    """
    Returns accuracy, precision, recall, f1 (macro)
    """
    probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    y_true = labels.detach().cpu().numpy()
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    return acc, prec, rec, f1, preds


# -----------------------------
# Training / Validation loop
# -----------------------------
def train(
    train_loader,
    val_loader,
    modality_input_dims,
    device,
    output_dir,
    epochs=100,
    lr=1e-4,
    cls_weight=1.0,
    surv_weight=1.0,
    patience=15,
    save_every=1
):
    os.makedirs(output_dir, exist_ok=True)
    model = UnifiedMultimodalModel(
        modality_input_dims=modality_input_dims,
        ae_latent_dim=64,
        transformer_cfg={"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512, "dropout": 0.3, "use_cls_token": True},
        num_classes=4
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        tic = time.time()
        model.train()
        train_losses = []
        train_cls_losses = []
        train_surv_losses = []

        for batch in train_loader:
            modalities, labels, durations, events = batch
            # Move modalities to device: modalities is list of tensors (B, feat_dim)
            modalities = [m.to(device) for m in modalities]
            labels = labels.to(device)
            durations = durations.to(device)
            events = events.to(device)

            optimizer.zero_grad()
            # add_noise_to_ae True to allow autoencoders add noise during training
            logits, log_risk, attn_weights, fused = model(modalities, modality_mask=None, add_noise_to_ae=True)

            # classification loss
            loss_cls = nn.CrossEntropyLoss()(logits, labels)
            # survival loss
            loss_surv = cox_partial_log_likelihood(log_risk, durations, events)

            loss = cls_weight * loss_cls + surv_weight * loss_surv

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_cls_losses.append(loss_cls.item())
            train_surv_losses.append(loss_surv.item())

        # Validation
        model.eval()
        val_losses = []
        val_cls_losses = []
        val_surv_losses = []
        all_val_labels = []
        all_val_preds = []
        val_durations_list = []
        val_events_list = []
        val_risks = []

        with torch.no_grad():
            for batch in val_loader:
                modalities, labels, durations, events = batch
                modalities = [m.to(device) for m in modalities]
                labels = labels.to(device)
                durations = durations.to(device)
                events = events.to(device)

                logits, log_risk, attn_weights, fused = model(modalities, modality_mask=None, add_noise_to_ae=False)

                loss_cls = nn.CrossEntropyLoss()(logits, labels)
                loss_surv = cox_partial_log_likelihood(log_risk, durations, events)
                loss = cls_weight * loss_cls + surv_weight * loss_surv

                val_losses.append(loss.item())
                val_cls_losses.append(loss_cls.item())
                val_surv_losses.append(loss_surv.item())

                # metrics
                acc, prec, rec, f1, preds = compute_classification_metrics(logits, labels)
                all_val_labels.extend(labels.detach().cpu().numpy().tolist())
                all_val_preds.extend(preds.tolist())

                # collect survival arrays for c-index
                val_durations_list.extend(durations.detach().cpu().numpy().tolist())
                val_events_list.extend(events.detach().cpu().numpy().tolist())
                val_risks.extend(torch.exp(log_risk).detach().cpu().numpy().reshape(-1).tolist())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses) if val_losses else float("nan")
        avg_train_cls = np.mean(train_cls_losses) if train_cls_losses else 0.0
        avg_train_surv = np.mean(train_surv_losses) if train_surv_losses else 0.0
        avg_val_cls = np.mean(val_cls_losses) if val_cls_losses else 0.0
        avg_val_surv = np.mean(val_surv_losses) if val_surv_losses else 0.0

        # Compute final metrics for validation set
        val_acc, val_prec, val_rec, val_f1 = 0.0, 0.0, 0.0, 0.0
        if len(all_val_preds) > 0:
            val_acc = accuracy_score(all_val_labels, all_val_preds)
            val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(all_val_labels, all_val_preds, average="macro", zero_division=0)

        # compute concordance index
        try:
            c_index = concordance_index(val_durations_list, -np.array(val_risks), val_events_list)
            # Note: many definitions expect higher risk -> shorter times, so we pass negative risk
        except Exception:
            c_index = float("nan")

        toc = time.time()
        epoch_time = toc - tic

        print(f"[Epoch {epoch:03d}] time: {epoch_time:.1f}s  train_loss: {avg_train_loss:.4f} (cls:{avg_train_cls:.4f}, surv:{avg_train_surv:.4f})  val_loss: {avg_val_loss:.4f} (cls:{avg_val_cls:.4f}, surv:{avg_val_surv:.4f})")
        print(f"             val_acc: {val_acc:.4f}  val_prec: {val_prec:.4f}  val_rec: {val_rec:.4f}  val_f1: {val_f1:.4f}  val_c-index: {c_index:.4f}")

        # scheduler step on validation loss
        scheduler.step(avg_val_loss)

        # checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print("  [INFO] Best model updated and saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping triggered after {epoch} epochs (no improvement in {patience} epochs).")
            break

    # load best weights before returning
    model.load_state_dict(best_model_wts)
    # final save
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    print(f"[INFO] Training completed. Best val loss: {best_val_loss:.4f}. Models saved in {output_dir}")

    return model


# -----------------------------
# CLI and main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train unified multi-modal Transformer model.")
    parser.add_argument("--train_npz", type=str, required=True, help="Path to train .npz (mod_*, labels, durations, events)")
    parser.add_argument("--val_npz", type=str, required=True, help="Path to validation .npz")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--surv_weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load datasets
    train_ds = MultimodalNPZDataset(args.train_npz)
    val_ds = MultimodalNPZDataset(args.val_npz)

    # modality input dims
    modality_input_dims = [arr.shape[1] for arr in train_ds.modalities]
    print(f"[INFO] Detected {len(modality_input_dims)} modalities with dims: {modality_input_dims}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_collate, num_workers=args.num_workers)

    # Train
    _ = train(
        train_loader=train_loader,
        val_loader=val_loader,
        modality_input_dims=modality_input_dims,
        device=device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        cls_weight=args.cls_weight,
        surv_weight=args.surv_weight,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
