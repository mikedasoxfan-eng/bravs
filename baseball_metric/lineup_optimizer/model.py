"""Phase 2: Lineup Value Model — predicts team runs from roster BRAVS composition.

Architecture: Two-stage model
  Stage 1: Team-level model that predicts R/G from roster composition
  Stage 2: Lineup permutation model that predicts ordering effects

Both run on GPU for fast inference during optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import logging

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LineupValueNetwork(nn.Module):
    """Neural network that predicts expected runs per game from lineup features.

    Input: lineup feature vector (hitting, baserunning, fielding, positional,
           depth, speed, power concentration, pitching quality, etc.)
    Output: predicted R/G with uncertainty (mean + log_var)
    """

    def __init__(self, n_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        # Mean and variance heads for uncertainty
        self.mean_head = nn.Linear(hidden // 2, 1)
        self.logvar_head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar

    def predict(self, x):
        """Predict with uncertainty (returns mean and std)."""
        self.eval()
        with torch.no_grad():
            mean, logvar = self(x)
            std = (logvar / 2).exp()
        return mean, std


class SlotInteractionModel(nn.Module):
    """Transformer that learns batting order interaction effects.

    Input: 9 player feature vectors (one per batting order slot)
    Output: predicted run value adjustment from ordering effects

    Uses self-attention to capture:
    - High-OBP hitter before power hitter (table-setter effect)
    - Speed threats affecting pitcher behavior for next batter
    - Lineup protection (pitcher can't pitch around hitter if next batter is also dangerous)
    """

    def __init__(self, player_dim: int = 8, n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.player_dim = player_dim

        # Positional encoding for batting order slots (1-9)
        self.slot_embedding = nn.Embedding(9, player_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=player_dim, nhead=n_heads, dim_feedforward=32,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output: single scalar (run adjustment from ordering)
        self.output = nn.Sequential(
            nn.Linear(player_dim * 9, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, player_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            player_features: (batch, 9, player_dim) — 9 batters' feature vectors

        Returns:
            (batch,) — predicted run adjustment from batting order
        """
        B = player_features.shape[0]

        # Add slot positional encoding
        slots = torch.arange(9, device=player_features.device).unsqueeze(0).expand(B, -1)
        slot_emb = self.slot_embedding(slots)
        x = player_features + slot_emb

        # Self-attention across lineup
        x = self.transformer(x)

        # Flatten and predict
        x = x.reshape(B, -1)
        return self.output(x).squeeze(-1)


def train_lineup_model(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 500,
    lr: float = 0.001,
    val_split: float = 0.15,
) -> LineupValueNetwork:
    """Train the lineup value network on team-season data.

    Uses Gaussian NLL loss to learn both mean prediction and uncertainty.
    """
    n = X.shape[0]
    n_val = int(n * val_split)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = LineupValueNetwork(X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        mean, logvar = model(X_train)
        # Gaussian negative log-likelihood
        loss = 0.5 * (logvar + (y_train - mean) ** 2 / logvar.exp()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_mean, val_logvar = model(X_val)
                val_loss = 0.5 * (val_logvar + (y_val - val_mean) ** 2 / val_logvar.exp()).mean()
                val_rmse = ((val_mean - y_val) ** 2).mean().sqrt()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            log.info("Epoch %d: train_loss=%.4f, val_loss=%.4f, val_rmse=%.3f R/G",
                     epoch + 1, loss.item(), val_loss.item(), val_rmse.item())

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model
