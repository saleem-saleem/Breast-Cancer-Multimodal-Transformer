import os
import json
import time
import argparse
from typing import List, Tuple

import numpy as np
import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score
from lifelines.utils import concordance_index

# Import your unified model and survival loss util
from models.unified_model import UnifiedMultimodalModel
from models.survival_head import cox_partial_log_likelihood

# -------------------------
# Dataset & collate (same format as train_model.py)
# -------------------------
class MultimodalNPZDataset(Dataset):
    def __init__(self, npz_path: str, modality_keys: List[str] = None):
        data = np.load(npz_path, allow_pickle=True)
        if modality_keys:
            self.modalities = [data[k].astype(np.float32) for k in modality_keys]
        else:
            mod_keys = sorted([k for k in data.keys() if k.startswith("mod_")])
            if len(mod_keys) == 0:
                raise ValueError(f"No modality arrays found in {npz_path}. Expected keys mod_0, mod_1, ...")
            self.modalities = [data[k].astype(np.float32) for k in mod_keys]

        if "labels" not in data or "durations" not in data or "events" not in data:
            raise ValueError(f"{npz_path} must contain 'labels', 'durations' and 'events' arrays.")
        self.labels = data["labels"].astype(np.int64)
        self.durations = data["durations"].astype(np.float32)
        self.events = data["events"].astype(np.float32)

        self.N = self.labels.shape[0]
        for arr in self.modalities:
            if arr.shape[0] != self.N:
                raise ValueError("All modality arrays must have same number of samples as labels.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        mods = [torch.from_numpy(arr[idx]) for arr in self.modalities]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        duration = torch.tensor(self.durations[idx], dtype=torch.float32)
        event = torch.tensor(self.events[idx], dtype=torch.float32)
        return mods, label, duration, event


def multimodal_collate(batch):
    num_modalities = len(batch[0][0])
    mods_per_modality = [[] for _ in range(num_modalities)]
    labels, durations, events = [], [], []
    for mods, lbl, dur, ev in batch:
        for i, m in enumerate(mods):
            mods_per_modality[i].append(m)
        labels.append(lbl)
        durations.append(dur)
        events.append(ev)
    mods_stacked = [torch.stack(lst, dim=0) for lst in mods_per_modality]
    labels = torch.stack(labels, dim=0)
    durations = torch.stack(durations, dim=0)
    events = torch.stack(events, dim=0)
    return mods_stacked, labels, durations, events

# -------------------------
# Utility functions
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_epoch(model, loader, device, cls_weight: float, surv_weight: float):
    """
    Run one evaluation pass and return average val loss and concordance-index.
    """
    model.eval()
    val_losses = []
    all_labels = []
    all_preds = []
    all_durations = []
    all_events = []
    all_risks = []

    with torch.no_grad():
        for modalities, labels, durations, events in loader:
            modalities = [m.to(device) for m in modalities]
            labels = labels.to(device)
            durations = durations.to(device)
            events = events.to(device)

            logits, log_risk, attn_weights, fused = model(modalities, modality_mask=None, add_noise_to_ae=False)

            loss_cls = nn.CrossEntropyLoss()(logits, labels)
            loss_surv = cox_partial_log_likelihood(log_risk, durations, events)
            loss = cls_weight * loss_cls + surv_weight * loss_surv

            val_losses.append(loss.item())

            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_durations.extend(durations.cpu().numpy().tolist())
            all_events.extend(events.cpu().numpy().tolist())
            all_risks.extend(torch.exp(log_risk).cpu().numpy().reshape(-1).tolist())

    avg_val_loss = np.mean(val_losses) if len(val_losses) > 0 else float("nan")
    try:
        # lifelines concordance_index expects higher predicted hazard -> shorter survival, so use negative risk if needed
        c_index = concordance_index(all_durations, -np.array(all_risks), all_events)
    except Exception:
        c_index = float("nan")

    # classification accuracy for monitoring
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else float("nan")
    return avg_val_loss, c_index, acc


# -------------------------
# Objective for Optuna
# -------------------------
def objective(trial: optuna.trial.Trial, args) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameter search space
    ae_latent_dim = trial.suggest_categorical("ae_latent_dim", [32, 64, 128])
    d_model = trial.suggest_categorical("d_model", [128, 256, 384])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    cls_hidden = trial.suggest_categorical("cls_hidden", [64, 128, 256])
    use_cls_token = trial.suggest_categorical("use_cls_token", [True, False])
    cls_weight = trial.suggest_float("cls_weight", 0.5, 2.0, step=0.25)
    surv_weight = trial.suggest_float("surv_weight", 0.5, 2.0, step=0.25)

    # Prepare datasets & loaders
    train_ds = MultimodalNPZDataset(args.train_npz)
    val_ds = MultimodalNPZDataset(args.val_npz)
    modality_input_dims = [arr.shape[1] for arr in train_ds.modalities]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate, num_workers=0)

    # Build model for this trial
    transformer_cfg = {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": max(256, d_model * 2),
        "dropout": dropout,
        "use_cls_token": use_cls_token,
        "max_seq_len": 32
    }

    model = UnifiedMultimodalModel(
        modality_input_dims=modality_input_dims,
        ae_latent_dim=ae_latent_dim,
        transformer_cfg=transformer_cfg,
        num_classes=args.num_classes,
        cls_hidden=cls_hidden,
        use_mlp_in_cls=True
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # small number of epochs per trial to speed up search
    epochs = args.epochs_per_trial
    best_val_cindex = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            modalities, labels, durations, events = batch
            modalities = [m.to(device) for m in modalities]
            labels = labels.to(device)
            durations = durations.to(device)
            events = events.to(device)

            optimizer.zero_grad()
            logits, log_risk, attn_weights, fused = model(modalities, modality_mask=None, add_noise_to_ae=True)

            loss_cls = nn.CrossEntropyLoss()(logits, labels)
            loss_surv = cox_partial_log_likelihood(log_risk, durations, events)
            loss = cls_weight * loss_cls + surv_weight * loss_surv

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate on validation set
        val_loss, val_cindex, val_acc = evaluate_epoch(model, val_loader, device, cls_weight, surv_weight)
        trial.report(val_cindex, epoch)

        # pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # keep track of best
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_epoch = epoch
            # save intermediate best model to trial-specific path
            trial.set_user_attr("best_epoch", best_epoch)
            # also save weights to temporary file for later retrieval if needed
            tmp_dir = args.tmp_dir
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"trial_{trial.number}_best.pt")
            torch.save(model.state_dict(), tmp_path)
            trial.set_user_attr("best_model_path", tmp_path)

        # optional logging per trial
        trial.set_user_attr("last_val_cindex", val_cindex)
        trial.set_user_attr("last_val_loss", val_loss)
        trial.set_user_attr("last_val_acc", val_acc)

    # Return negative because Optuna by default minimizes? (we configure study to maximize)
    return best_val_cindex


# -------------------------
# Main: run study
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization using Optuna")
    parser.add_argument("--train_npz", type=str, required=True, help="Path to training .npz")
    parser.add_argument("--val_npz", type=str, required=True, help="Path to validation .npz")
    parser.add_argument("--num_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs_per_trial", type=int, default=8, help="Epochs per trial (small for speed during search)")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of recurrence classes")
    parser.add_argument("--study_name", type=str, default="mm_transformer_hp_search", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--tmp_dir", type=str, default="optuna_tmp", help="Directory to save trial temporary models")
    parser.add_argument("--output_dir", type=str, default="optuna_results", help="Directory to save best results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    direction = "maximize"  # maximize C-index
    study = optuna.create_study(direction=direction, study_name=args.study_name, storage=args.storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1, interval_steps=1)
                                )

    # objective wrapper with args
    func = lambda trial: objective(trial, args)

    print(f"[INFO] Starting Optuna study ({args.num_trials} trials)...")
    try:
        study.optimize(func, n_trials=args.num_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("[WARN] Optimization interrupted by user.")

    print("== Study stats ==")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", sum(1 for t in study.trials if t.state == TrialState.PRUNED))
    print("  Number of complete trials: ", sum(1 for t in study.trials if t.state == TrialState.COMPLETE))

    # Best trial
    best = study.best_trial
    print("Best trial:")
    print(f"  Value (val c-index): {best.value}")
    print("  Params: ")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # save best trial config
    best_cfg_path = os.path.join(args.output_dir, "best_trial_params.json")
    with open(best_cfg_path, "w") as f:
        json.dump({"value": float(best.value), "params": best.params}, f, indent=2)
    print(f"[INFO] Best trial params saved to {best_cfg_path}")

    # copy best model weights (if recorded)
    best_model_path = best.user_attrs.get("best_model_path", None)
    if best_model_path and os.path.exists(best_model_path):
        final_model_path = os.path.join(args.output_dir, "best_model_from_optuna.pt")
        torch.save(torch.load(best_model_path, map_location="cpu"), final_model_path)
        print(f"[INFO] Best model checkpoint copied to {final_model_path}")
    else:
        print("[WARN] Best model path not found among trial user attributes.")

    # optionally save full study
    study_path = os.path.join(args.output_dir, "optuna_study.pkl")
    try:
        import joblib
        joblib.dump(study, study_path)
        print(f"[INFO] Optuna study object saved to {study_path}")
    except Exception as e:
        print(f"[WARN] Could not save study object: {e}")

    print("[DONE] Hyperparameter optimization finished.")


if __name__ == "__main__":
    main()
