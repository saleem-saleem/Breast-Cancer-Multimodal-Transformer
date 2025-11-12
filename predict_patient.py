import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from models.unified_model import UnifiedMultimodalModel


# ============================================================
# 🧩 1. Load and Prepare Patient Data
# ============================================================
def load_patient_data(input_path):
    """
    Load patient data from .npz or JSON file.

    Returns:
        modalities: list of torch tensors [ (1, dim1), (1, dim2), ... ]
    """
    if input_path.endswith(".npz"):
        data = np.load(input_path, allow_pickle=True)
        mod_keys = sorted([k for k in data.keys() if k.startswith("mod_")])
        modalities = [torch.tensor(data[k][None, :], dtype=torch.float32) for k in mod_keys]
        print(f"[INFO] Loaded {len(modalities)} modalities from NPZ: {mod_keys}")

    elif input_path.endswith(".json"):
        with open(input_path, "r") as f:
            patient = json.load(f)
        # Expect format: {"mod_0": [...], "mod_1": [...], ...}
        mod_keys = sorted([k for k in patient.keys() if k.startswith("mod_")])
        modalities = [torch.tensor(patient[k], dtype=torch.float32).unsqueeze(0) for k in mod_keys]
        print(f"[INFO] Loaded {len(modalities)} modalities from JSON: {mod_keys}")

    else:
        raise ValueError("Unsupported file format. Use .npz or .json")

    return modalities


# ============================================================
# 🧠 2. Load Model
# ============================================================
def load_model(model_path, modality_input_dims, num_classes=4, device=None):
    """
    Loads the trained UnifiedMultimodalModel.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UnifiedMultimodalModel(
        modality_input_dims=modality_input_dims,
        ae_latent_dim=64,
        transformer_cfg={
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.3,
            "use_cls_token": True
        },
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Model loaded from {model_path}")
    return model


# ============================================================
# 📈 3. Run Prediction
# ============================================================
def predict_patient(model, modalities, modality_names=None, device=None, visualize_attention=True):
    """
    Predicts recurrence class probabilities and survival risk for a patient.

    Args:
        model: trained UnifiedMultimodalModel
        modalities: list of tensors [(1, dim_i), ...]
        modality_names: list of modality labels (for visualization)
        visualize_attention: if True, plots attention weights heatmap
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = [m.to(device) for m in modalities]

    with torch.no_grad():
        logits, log_risk, attn_weights, fused = model(modalities)

    probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    predicted_class = np.argmax(probs)
    risk_score = torch.exp(log_risk).cpu().numpy().flatten()[0]
    attn_weights = attn_weights.cpu().numpy().flatten()

    print("\n🩺 Patient Prediction Results")
    print("-" * 40)
    for i, p in enumerate(probs):
        print(f"Class {i}: {p:.4f}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Survival Risk Score (exp(log_risk)): {risk_score:.4f}")
    print(f"Modality Attention Weights: {attn_weights}")

    if visualize_attention:
        modality_names = modality_names or [f"Modality {i+1}" for i in range(len(attn_weights))]
        plot_attention_weights(attn_weights, modality_names)

    return predicted_class, probs, risk_score, attn_weights


# ============================================================
# 💠 4. Plot Modality Attention
# ============================================================
def plot_attention_weights(attn_weights, modality_names):
    """
    Plots attention weights for each modality as a bar chart.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=modality_names, y=attn_weights, palette="Blues_d", ax=ax)
    plt.title("Modality Attention Weights (Patient-Specific)")
    plt.ylabel("Weight")
    plt.xlabel("Modality")
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/patient_attention_weights.png", bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved attention weight plot to results/plots/patient_attention_weights.png")


# ============================================================
# 🚀 5. Main CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Predict patient recurrence & survival risk using trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--patient_data", type=str, required=True, help="Path to patient data (.npz or .json)")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of recurrence classes")
    parser.add_argument("--modality_names", type=str, nargs="+", default=["Clinical", "Genomic", "Lifestyle"])
    parser.add_argument("--no_visuals", action="store_true", help="Disable attention visualization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load patient data
    modalities = load_patient_data(args.patient_data)
    modality_input_dims = [m.shape[1] for m in modalities]

    # Load trained model
    model = load_model(args.model_path, modality_input_dims, num_classes=args.num_classes, device=device)

    # Run prediction
    predict_patient(
        model,
        modalities,
        modality_names=args.modality_names,
        device=device,
        visualize_attention=not args.no_visuals
    )


if __name__ == "__main__":
    main()
