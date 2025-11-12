from typing import List, Optional, Sequence, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

# Import components (assumes this file lives in the same package as the modules)
from models.autoencoder_module import DenoisingAutoencoder
from models.transformer_module import ModalityAttention, TransformerBackbone
from models.classification_head import ClassificationHead, classification_loss
from models.survival_head import SurvivalHead, cox_partial_log_likelihood

# -----------------------------------------------------------------------------
# Unified Model
# -----------------------------------------------------------------------------
class UnifiedMultimodalModel(nn.Module):
    """
    Unified model for multi-modal recurrence classification + survival prediction.

    Args:
        modality_input_dims: list of ints, input dims for each modality (e.g., [clin_dim, genom_dim, life_dim])
        ae_latent_dim: int, latent dim of each modality autoencoder (same for all by default)
        transformer_cfg: dict, config for TransformerBackbone (keys: d_model, nhead, num_layers, ...)
        num_classes: int, number of recurrence classes
        dropout: float, dropout used in heads
    """
    def __init__(
        self,
        modality_input_dims: Sequence[int],
        ae_latent_dim: int = 128,
        transformer_cfg: Optional[Dict[str, Any]] = None,
        num_classes: int = 4,
        cls_hidden: int = 128,
        use_mlp_in_cls: bool = True,
        device: Optional[torch.device] = None
    ):
        super(UnifiedMultimodalModel, self).__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.num_modalities = len(modality_input_dims)
        self.ae_latent_dim = ae_latent_dim

        # Create one autoencoder per modality
        self.autoencoders = nn.ModuleList([
            DenoisingAutoencoder(input_dim=d, latent_dim=ae_latent_dim)
            for d in modality_input_dims
        ])

        # Modality attention (works on latent_dim)
        self.modality_attention = ModalityAttention(embed_dim=ae_latent_dim, hidden_dim=max(64, ae_latent_dim // 2))

        # Transformer backbone configuration
        transformer_cfg = transformer_cfg or {}
        # ensure latent_dim is provided; other defaults set inside TransformerBackbone
        self.transformer = TransformerBackbone(
            latent_dim=ae_latent_dim,
            d_model=transformer_cfg.get("d_model", 256),
            nhead=transformer_cfg.get("nhead", 8),
            num_layers=transformer_cfg.get("num_layers", 4),
            dim_feedforward=transformer_cfg.get("dim_feedforward", 512),
            dropout=transformer_cfg.get("dropout", 0.3),
            use_cls_token=transformer_cfg.get("use_cls_token", True),
            max_seq_len=transformer_cfg.get("max_seq_len", 32)
        )

        d_model = transformer_cfg.get("d_model", 256)

        # Downstream heads
        self.class_head = ClassificationHead(input_dim=d_model, num_classes=num_classes,
                                             hidden_dim=cls_hidden, dropout=0.3, use_mlp=use_mlp_in_cls)
        self.surv_head = SurvivalHead(input_dim=d_model, dropout=0.2)

        # Put model on device
        self.to(self.device)

    def forward(
        self,
        modalities: List[torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        add_noise_to_ae: bool = False
    ):
        """
        Forward pass.

        Args:
            modalities: list of tensors, each with shape (B, input_dim_i). Length must equal num_modalities.
            modality_mask: optional tensor (B, M) with 1 for present modality, 0 for missing. If None, all present.
            add_noise_to_ae: if True, autoencoders will add training noise (useful during training)
        Returns:
            logits: (B, num_classes) classification logits
            log_risk: (B, 1) Cox log-risk scores
            attn_weights: (B, M) modality attention weights
            fused_repr: (B, d_model) fused embedding from transformer (useful for debugging or downstream)
        """
        B = modalities[0].shape[0]
        if len(modalities) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(modalities)}")

        # Move inputs to device and ensure float
        modalities = [m.to(self.device).float() for m in modalities]

        # Encode each modality via its autoencoder -> latent (B, latent_dim)
        latents = []
        for i, (ae, x) in enumerate(zip(self.autoencoders, modalities)):
            # DenoisingAutoencoder.forward returns (reconstructed, z)
            ae.train() if add_noise_to_ae else ae.eval()
            with torch.set_grad_enabled(self.training):  # allow grads during training; no grads if eval
                reconstructed, z = ae(x, add_noise=add_noise_to_ae)
            latents.append(z)

        # Stack latents -> (B, M, latent_dim)
        modality_latents = torch.stack(latents, dim=1)  # (B, M, latent_dim)

        # If modality_mask is None, assume all present
        if modality_mask is None:
            modality_mask = torch.ones(B, self.num_modalities, dtype=torch.bool, device=self.device)
        else:
            modality_mask = modality_mask.to(self.device).bool()

        # Compute modality attention weights and weighted embeddings (on latent space)
        weighted_latents, attn_weights = self.modality_attention(modality_latents)  # weighted_latents: (B,M,D)

        # Optional: zero out missing modalities (attention already multiplies, but enforce mask)
        mask_float = modality_mask.unsqueeze(-1).float()  # (B, M, 1)
        weighted_latents = weighted_latents * mask_float

        # Feed to Transformer backbone
        fused_repr, transformer_out = self.transformer(weighted_latents, modality_mask=modality_mask)  # fused: (B, d_model)

        # Heads
        logits = self.class_head(fused_repr)            # (B, num_classes)
        log_risk = self.surv_head(fused_repr)           # (B, 1)

        return logits, log_risk, attn_weights, fused_repr

    # -------------------------
    # Loss / Training convenience
    # -------------------------
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        log_risk: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
        cls_weight: float = 1.0,
        surv_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss (classification + Cox survival).
        Returns dict with total loss and components.
        """
        labels = labels.to(self.device).long()
        durations = durations.to(self.device).float()
        events = events.to(self.device).float()

        # Classification loss (weighted cross-entropy if class_weights provided)
        if class_weights is not None:
            class_weights = class_weights.to(self.device).float()
        loss_cls = classification_loss(logits, labels, class_weights)

        # Cox partial log-likelihood (negative)
        loss_surv = cox_partial_log_likelihood(log_risk, durations, events)

        total_loss = cls_weight * loss_cls + surv_weight * loss_surv

        return {"loss": total_loss, "loss_cls": loss_cls.detach(), "loss_surv": loss_surv.detach()}

    # -------------------------
    # Save / Load helpers
    # -------------------------
    def save(self, path: str):
        """
        Save model state dict to path.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None):
        """
        Load state dict from path.
        """
        self.load_state_dict(torch.load(path, map_location=map_location or self.device))

# -----------------------------------------------------------------------------
# Example / Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    # Suppose we have 3 modalities with different input dims
    modality_dims = [20, 100, 10]   # clinical (20), genomic (100), lifestyle (10)
    batch_size = 16

    # Create synthetic data
    X_mods = [
        torch.tensor(np.random.randn(200, d), dtype=torch.float32)
        for d in modality_dims
    ]
    # Create labels and survival targets
    labels = torch.tensor(np.random.randint(0, 4, size=(200,)), dtype=torch.long)
    durations = torch.tensor(np.random.randint(1, 1000, size=(200,)), dtype=torch.float32)
    events = torch.tensor(np.random.randint(0, 2, size=(200,)), dtype=torch.float32)

    # Create DataLoader that yields lists of modality tensors per batch
    dataset = TensorDataset(X_mods[0], X_mods[1], X_mods[2], labels, durations, events)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate unified model
    model = UnifiedMultimodalModel(
        modality_input_dims=modality_dims,
        ae_latent_dim=64,
        transformer_cfg={"d_model": 128, "nhead": 4, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.1, "use_cls_token": True},
        num_classes=4,
        cls_hidden=64,
    )

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Single epoch training loop (demo)
    for batch in loader:
        # Unpack batch
        mod1, mod2, mod3, y, dur, ev = batch
        modalities = [mod1, mod2, mod3]
        # optional modality mask (all present here)
        modality_mask = torch.ones(mod1.shape[0], len(modalities), dtype=torch.bool)

        logits, log_risk, attn_weights, fused = model(modalities, modality_mask=modality_mask, add_noise_to_ae=True)

        losses = model.compute_loss(
            logits=logits,
            labels=y,
            log_risk=log_risk,
            durations=dur,
            events=ev,
            cls_weight=1.0,
            surv_weight=1.0,
            class_weights=None
        )

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        print(f"Batch loss: {losses['loss'].item():.4f}, cls: {losses['loss_cls']:.4f}, surv: {losses['loss_surv']:.4f}")
        # show attn weights for first sample
        print("attn (first sample):", attn_weights[0].detach().cpu().numpy())
        break

    print("Smoke test completed.")
