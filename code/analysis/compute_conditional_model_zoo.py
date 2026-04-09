#!/usr/bin/env python3
"""Conditional return model zoo for 0DTE structures.

Implements walk-forward OOS evaluation for several model classes with flexible
feature/label scaling:
- Cross-sectional scaling by date for selected instrument features.
- Time-series scaling (train-window only) for selected feature groups.
- Optional label scaling (train-window z-score or robust z-score).

Outputs summary CSV + LaTeX table + predictions parquet.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression, RidgeCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

# Silence non-actionable runtime noise in constrained sandbox environments.
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


from _paths import get_project_root, get_data_dir, get_tables_dir  # noqa: E402

# Reuse helpers from strict OOS protocol script.
from compute_conditional_oos_protocol import (  # noqa: E402
    annualized_sharpe,
    build_feature_frame,
    calibr_slope,
    get_legs,
)
from moneyness_selection import (  # noqa: E402
    RepresentativeSelectionConfig,
    apply_representative_filter,
    choose_representative_moneyness,
)


@dataclass(frozen=True)
class ScalingSpec:
    name: str
    feature_space: str
    ts_mode: str
    y_mode: str


FEATURE_SETS = ("baseline", "gex", "flow", "liquidity", "all")


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _activation_layer(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "logistic":
        return nn.Sigmoid()
    if key == "gelu":
        return nn.GELU()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation: {name}")


class TorchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        activation: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(_activation_layer(activation))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _softplus_inverse(x: float) -> float:
    x = float(max(x, 1e-12))
    if x > 20.0:
        return x
    return float(np.log(np.expm1(x)))


class TorchLinearRidge(nn.Module):
    """Single-layer linear model with external ridge-alpha control."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class TorchLinearRidgeRegressor(BaseEstimator, RegressorMixin):
    """Linear PyTorch regressor with trainable ridge penalty alpha."""

    def __init__(
        self,
        init_alpha: float = 1.0,
        min_alpha: float = 1e-8,
        alpha_prior_strength: float = 1e-3,
        lr: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 120,
        patience: int = 15,
        validation_fraction: float = 0.1,
        min_delta: float = 1e-5,
        clip_grad: float = 1.0,
        random_state: int = 0,
        device: str = "cpu",
        verbose: int = 0,
    ) -> None:
        self.init_alpha = init_alpha
        self.min_alpha = min_alpha
        self.alpha_prior_strength = alpha_prior_strength
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.min_delta = min_delta
        self.clip_grad = clip_grad
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        n = int(X_np.shape[0])
        if n == 0:
            raise ValueError("TorchLinearRidgeRegressor.fit received empty training data")

        _set_torch_seed(int(self.random_state))
        use_device = self.device
        if use_device.startswith("cuda") and (not torch.cuda.is_available()):
            use_device = "cpu"
        self.device_ = torch.device(use_device)

        idx = np.arange(n)
        rng = np.random.default_rng(int(self.random_state))
        rng.shuffle(idx)
        val_n = int(max(0, min(n - 1, round(n * float(self.validation_fraction)))))
        tr_n = n - val_n
        tr_idx = idx[:tr_n]
        va_idx = idx[tr_n:] if val_n > 0 else np.array([], dtype=int)

        X_tr = torch.from_numpy(X_np[tr_idx])
        y_tr = torch.from_numpy(y_np[tr_idx])
        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(min(max(32, self.batch_size), max(32, tr_n))),
            shuffle=True,
            drop_last=False,
        )

        if val_n > 0:
            X_va = torch.from_numpy(X_np[va_idx])
            y_va = torch.from_numpy(y_np[va_idx])
            val_ds = TensorDataset(X_va, y_va)
            val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, drop_last=False)
        else:
            val_loader = None

        self.model_ = TorchLinearRidge(input_dim=int(X_np.shape[1])).to(self.device_)

        alpha0 = float(max(self.init_alpha, self.min_alpha))
        raw_alpha0 = _softplus_inverse(alpha0 - float(self.min_alpha))
        self.log_alpha_param_ = nn.Parameter(torch.tensor(raw_alpha0, dtype=torch.float32, device=self.device_))
        alpha0_tensor = torch.tensor(raw_alpha0, dtype=torch.float32, device=self.device_)

        optimizer = torch.optim.Adam(
            [*self.model_.parameters(), self.log_alpha_param_],
            lr=float(self.lr),
        )
        loss_fn = nn.MSELoss()

        best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
        best_log_alpha = self.log_alpha_param_.detach().cpu().clone()
        best_metric = np.inf
        bad_epochs = 0

        for epoch in range(int(self.max_epochs)):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                mse = loss_fn(pred, yb)
                alpha = F.softplus(self.log_alpha_param_) + float(self.min_alpha)
                ridge_pen = alpha * self.model_.linear.weight.pow(2).mean()
                alpha_prior = float(self.alpha_prior_strength) * (self.log_alpha_param_ - alpha0_tensor).pow(2)
                loss = mse + ridge_pen + alpha_prior
                loss.backward()
                if self.clip_grad and self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(
                        [*self.model_.parameters(), self.log_alpha_param_],
                        max_norm=float(self.clip_grad),
                    )
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                if val_loader is not None:
                    vals = []
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        vals.append(loss_fn(self.model_(xb), yb).item())
                    metric = float(np.mean(vals)) if vals else np.inf
                else:
                    train_vals = []
                    for xb, yb in train_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        train_vals.append(loss_fn(self.model_(xb), yb).item())
                    metric = float(np.mean(train_vals)) if train_vals else np.inf

            if metric + float(self.min_delta) < best_metric:
                best_metric = metric
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                best_log_alpha = self.log_alpha_param_.detach().cpu().clone()
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break

            if self.verbose and (epoch + 1) % 20 == 0:
                alpha_now = float((F.softplus(self.log_alpha_param_) + float(self.min_alpha)).item())
                print(
                    f"[TorchLinearRidgeRegressor] epoch={epoch + 1} "
                    f"metric={metric:.6f} alpha={alpha_now:.6f}"
                )

        self.model_.load_state_dict(best_state)
        self.log_alpha_param_.data = best_log_alpha.to(self.device_)
        self.model_.eval()
        self.alpha_ = float((F.softplus(self.log_alpha_param_) + float(self.min_alpha)).item())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchLinearRidgeRegressor is not fitted")
        X_np = np.asarray(X, dtype=np.float32)
        xb = torch.from_numpy(X_np).to(self.device_)
        with torch.no_grad():
            pred = self.model_(xb).detach().cpu().numpy()
        return pred.astype(float)


class TorchCNN1D(nn.Module):
    """Compact 1D CNN for tabular features interpreted as a sequence."""

    def __init__(
        self,
        input_dim: int,
        channels: tuple[int, int] = (32, 16),
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2 = int(channels[0]), int(channels[1])
        k = int(max(1, kernel_size))
        pad = k // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k, padding=pad),
            _activation_layer(activation),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv1d(c1, c2, kernel_size=k, padding=pad),
            _activation_layer(activation),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c2, 1)
        self.input_dim = int(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, 1, F]
        z = self.features(x)
        z = self.pool(z).squeeze(-1)  # [B, C]
        return self.head(z).squeeze(-1)


class TorchCNNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible 1D CNN regressor."""

    def __init__(
        self,
        channels: tuple[int, int] = (32, 16),
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.05,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 120,
        patience: int = 15,
        validation_fraction: float = 0.1,
        min_delta: float = 1e-5,
        clip_grad: float = 1.0,
        random_state: int = 0,
        device: str = "cpu",
        verbose: int = 0,
    ) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.min_delta = min_delta
        self.clip_grad = clip_grad
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        n = int(X_np.shape[0])
        if n == 0:
            raise ValueError("TorchCNNRegressor.fit received empty training data")

        _set_torch_seed(int(self.random_state))
        use_device = self.device
        if use_device.startswith("cuda") and (not torch.cuda.is_available()):
            use_device = "cpu"
        self.device_ = torch.device(use_device)

        idx = np.arange(n)
        rng = np.random.default_rng(int(self.random_state))
        rng.shuffle(idx)
        val_n = int(max(0, min(n - 1, round(n * float(self.validation_fraction)))))
        tr_n = n - val_n
        tr_idx = idx[:tr_n]
        va_idx = idx[tr_n:] if val_n > 0 else np.array([], dtype=int)

        X_tr = torch.from_numpy(X_np[tr_idx])
        y_tr = torch.from_numpy(y_np[tr_idx])
        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(min(max(32, self.batch_size), max(32, tr_n))),
            shuffle=True,
            drop_last=False,
        )

        if val_n > 0:
            X_va = torch.from_numpy(X_np[va_idx])
            y_va = torch.from_numpy(y_np[va_idx])
            val_ds = TensorDataset(X_va, y_va)
            val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, drop_last=False)
        else:
            val_loader = None

        self.model_ = TorchCNN1D(
            input_dim=int(X_np.shape[1]),
            channels=tuple(int(c) for c in self.channels),
            kernel_size=int(self.kernel_size),
            activation=self.activation,
            dropout=float(self.dropout),
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.MSELoss()

        best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
        best_metric = np.inf
        bad_epochs = 0

        for epoch in range(int(self.max_epochs)):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                if self.clip_grad and self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=float(self.clip_grad))
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                if val_loader is not None:
                    vals = []
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        vals.append(loss_fn(self.model_(xb), yb).item())
                    metric = float(np.mean(vals)) if vals else np.inf
                else:
                    train_vals = []
                    for xb, yb in train_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        train_vals.append(loss_fn(self.model_(xb), yb).item())
                    metric = float(np.mean(train_vals)) if train_vals else np.inf

            if metric + float(self.min_delta) < best_metric:
                best_metric = metric
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"[TorchCNNRegressor] epoch={epoch + 1} metric={metric:.6f}")

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchCNNRegressor is not fitted")
        X_np = np.asarray(X, dtype=np.float32)
        xb = torch.from_numpy(X_np).to(self.device_)
        with torch.no_grad():
            pred = self.model_(xb).detach().cpu().numpy()
        return pred.astype(float)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible PyTorch MLP regressor."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 120,
        patience: int = 15,
        validation_fraction: float = 0.1,
        min_delta: float = 1e-5,
        clip_grad: float = 1.0,
        random_state: int = 0,
        device: str = "cpu",
        verbose: int = 0,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.min_delta = min_delta
        self.clip_grad = clip_grad
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        n = int(X_np.shape[0])
        if n == 0:
            raise ValueError("TorchMLPRegressor.fit received empty training data")

        _set_torch_seed(int(self.random_state))
        use_device = self.device
        if use_device.startswith("cuda") and (not torch.cuda.is_available()):
            use_device = "cpu"
        self.device_ = torch.device(use_device)

        idx = np.arange(n)
        rng = np.random.default_rng(int(self.random_state))
        rng.shuffle(idx)
        val_n = int(max(0, min(n - 1, round(n * float(self.validation_fraction)))))
        tr_n = n - val_n
        tr_idx = idx[:tr_n]
        va_idx = idx[tr_n:] if val_n > 0 else np.array([], dtype=int)

        X_tr = torch.from_numpy(X_np[tr_idx])
        y_tr = torch.from_numpy(y_np[tr_idx])
        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(min(max(32, self.batch_size), max(32, tr_n))),
            shuffle=True,
            drop_last=False,
        )

        if val_n > 0:
            X_va = torch.from_numpy(X_np[va_idx])
            y_va = torch.from_numpy(y_np[va_idx])
            val_ds = TensorDataset(X_va, y_va)
            val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, drop_last=False)
        else:
            val_loader = None

        self.model_ = TorchMLP(
            input_dim=int(X_np.shape[1]),
            hidden_layer_sizes=tuple(int(h) for h in self.hidden_layer_sizes),
            activation=self.activation,
            dropout=float(self.dropout),
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.MSELoss()

        best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
        best_metric = np.inf
        bad_epochs = 0

        for epoch in range(int(self.max_epochs)):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                if self.clip_grad and self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=float(self.clip_grad))
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                if val_loader is not None:
                    vals = []
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        vals.append(loss_fn(self.model_(xb), yb).item())
                    metric = float(np.mean(vals)) if vals else np.inf
                else:
                    train_vals = []
                    for xb, yb in train_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        train_vals.append(loss_fn(self.model_(xb), yb).item())
                    metric = float(np.mean(train_vals)) if train_vals else np.inf

            if metric + float(self.min_delta) < best_metric:
                best_metric = metric
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"[TorchMLPRegressor] epoch={epoch + 1} metric={metric:.6f}")

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMLPRegressor is not fitted")
        X_np = np.asarray(X, dtype=np.float32)
        xb = torch.from_numpy(X_np).to(self.device_)
        with torch.no_grad():
            pred = self.model_(xb).detach().cpu().numpy()
        return pred.astype(float)


class TorchBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible PyTorch binary classifier for tabular features."""

    def __init__(
        self,
        model_type: str = "mlp",
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        channels: tuple[int, int] = (32, 16),
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.05,
        init_alpha: float = 1.0,
        min_alpha: float = 1e-8,
        alpha_prior_strength: float = 1e-3,
        class_weight_balance: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 120,
        patience: int = 15,
        validation_fraction: float = 0.1,
        min_delta: float = 1e-5,
        clip_grad: float = 1.0,
        random_state: int = 0,
        device: str = "cpu",
        verbose: int = 0,
    ) -> None:
        self.model_type = model_type
        self.hidden_layer_sizes = hidden_layer_sizes
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
        self.init_alpha = init_alpha
        self.min_alpha = min_alpha
        self.alpha_prior_strength = alpha_prior_strength
        self.class_weight_balance = class_weight_balance
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.min_delta = min_delta
        self.clip_grad = clip_grad
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _build_model(self, input_dim: int) -> nn.Module:
        key = str(self.model_type).lower()
        if key == "mlp":
            return TorchMLP(
                input_dim=int(input_dim),
                hidden_layer_sizes=tuple(int(h) for h in self.hidden_layer_sizes),
                activation=self.activation,
                dropout=float(self.dropout),
            )
        if key == "cnn":
            return TorchCNN1D(
                input_dim=int(input_dim),
                channels=tuple(int(c) for c in self.channels),
                kernel_size=int(self.kernel_size),
                activation=self.activation,
                dropout=float(self.dropout),
            )
        if key == "linear_ridge":
            return TorchLinearRidge(input_dim=int(input_dim))
        raise ValueError(f"Unsupported TorchBinaryClassifier model_type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = (np.asarray(y, dtype=np.float32).reshape(-1) > 0.5).astype(np.float32)
        n = int(X_np.shape[0])
        if n == 0:
            raise ValueError("TorchBinaryClassifier.fit received empty training data")

        _set_torch_seed(int(self.random_state))
        use_device = self.device
        if use_device.startswith("cuda") and (not torch.cuda.is_available()):
            use_device = "cpu"
        self.device_ = torch.device(use_device)

        idx = np.arange(n)
        rng = np.random.default_rng(int(self.random_state))
        rng.shuffle(idx)
        val_n = int(max(0, min(n - 1, round(n * float(self.validation_fraction)))))
        tr_n = n - val_n
        tr_idx = idx[:tr_n]
        va_idx = idx[tr_n:] if val_n > 0 else np.array([], dtype=int)

        X_tr = torch.from_numpy(X_np[tr_idx])
        y_tr = torch.from_numpy(y_np[tr_idx])
        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(min(max(32, self.batch_size), max(32, tr_n))),
            shuffle=True,
            drop_last=False,
        )

        if val_n > 0:
            X_va = torch.from_numpy(X_np[va_idx])
            y_va = torch.from_numpy(y_np[va_idx])
            val_ds = TensorDataset(X_va, y_va)
            val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, drop_last=False)
        else:
            val_loader = None

        self.model_ = self._build_model(input_dim=int(X_np.shape[1])).to(self.device_)

        opt_params = list(self.model_.parameters())
        use_linear_ridge = str(self.model_type).lower() == "linear_ridge"
        alpha0_tensor = None
        if use_linear_ridge:
            alpha0 = float(max(self.init_alpha, self.min_alpha))
            raw_alpha0 = _softplus_inverse(alpha0 - float(self.min_alpha))
            self.log_alpha_param_ = nn.Parameter(torch.tensor(raw_alpha0, dtype=torch.float32, device=self.device_))
            alpha0_tensor = torch.tensor(raw_alpha0, dtype=torch.float32, device=self.device_)
            opt_params = [*opt_params, self.log_alpha_param_]
        else:
            self.log_alpha_param_ = None

        optimizer = torch.optim.AdamW(
            opt_params,
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

        pos_w = 1.0
        if bool(self.class_weight_balance):
            pos = float(y_np[tr_idx].sum())
            neg = float(tr_n - pos)
            if pos > 0 and neg > 0:
                pos_w = float(np.clip(neg / pos, 0.25, 4.0))
        pos_w_t = torch.tensor(pos_w, dtype=torch.float32, device=self.device_)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)

        best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
        best_log_alpha = self.log_alpha_param_.detach().cpu().clone() if self.log_alpha_param_ is not None else None
        best_metric = np.inf
        bad_epochs = 0

        for epoch in range(int(self.max_epochs)):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                bce = loss_fn(logits, yb)

                if use_linear_ridge:
                    alpha = F.softplus(self.log_alpha_param_) + float(self.min_alpha)
                    ridge_pen = alpha * self.model_.linear.weight.pow(2).mean()
                    alpha_prior = float(self.alpha_prior_strength) * (self.log_alpha_param_ - alpha0_tensor).pow(2)
                    loss = bce + ridge_pen + alpha_prior
                else:
                    loss = bce

                loss.backward()
                if self.clip_grad and self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(opt_params, max_norm=float(self.clip_grad))
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                if val_loader is not None:
                    vals = []
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        vals.append(loss_fn(self.model_(xb), yb).item())
                    metric = float(np.mean(vals)) if vals else np.inf
                else:
                    train_vals = []
                    for xb, yb in train_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        train_vals.append(loss_fn(self.model_(xb), yb).item())
                    metric = float(np.mean(train_vals)) if train_vals else np.inf

            if metric + float(self.min_delta) < best_metric:
                best_metric = metric
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                if self.log_alpha_param_ is not None:
                    best_log_alpha = self.log_alpha_param_.detach().cpu().clone()
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break

            if self.verbose and (epoch + 1) % 20 == 0:
                if use_linear_ridge:
                    alpha_now = float((F.softplus(self.log_alpha_param_) + float(self.min_alpha)).item())
                    print(f"[TorchBinaryClassifier] epoch={epoch + 1} metric={metric:.6f} alpha={alpha_now:.6f}")
                else:
                    print(f"[TorchBinaryClassifier] epoch={epoch + 1} metric={metric:.6f}")

        self.model_.load_state_dict(best_state)
        if self.log_alpha_param_ is not None and best_log_alpha is not None:
            self.log_alpha_param_.data = best_log_alpha.to(self.device_)
            self.alpha_ = float((F.softplus(self.log_alpha_param_) + float(self.min_alpha)).item())
        self.model_.eval()
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchBinaryClassifier is not fitted")
        X_np = np.asarray(X, dtype=np.float32)
        xb = torch.from_numpy(X_np).to(self.device_)
        with torch.no_grad():
            logits = self.model_(xb).detach().cpu().numpy()
        return logits.astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1]).astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conditional model-zoo OOS backtests.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--task",
        choices=["regression", "binary"],
        default="regression",
        help="Prediction task: continuous returns (regression) or binary direction (binary).",
    )
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--rolling-window", type=int, default=252)
    parser.add_argument("--protocols", nargs="+", default=["expanding", "rolling"])
    parser.add_argument("--refit-every", type=int, default=1, help="Refit model every N OOS dates.")
    parser.add_argument(
        "--decision-mode",
        choices=["hard", "soft"],
        default="hard",
        help=(
            "How probabilities are converted to positions for economic PNL. "
            "hard = +/-1 sign, soft = continuous weight in [-1,1]."
        ),
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for long/short directional decision (default 0.5).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional explicit model list. If omitted, defaults depend on --task. "
            "Regression examples: ridge elastic_net lgbm rf xgb catboost nn_linear_ridge nn_cnn nn_relu. "
            "Binary examples: logit ridge_logit elastic_net_logit lgbm_clf rf_clf et_clf "
            "xgb_clf catboost_clf nn_linear_ridge_clf nn_cnn_clf nn_relu_clf"
        ),
    )
    parser.add_argument(
        "--scalings",
        nargs="+",
        default=["ts_raw_yz", "cs_tsinstr_yz", "cs_tsall_yz", "cs_tsall_yraw", "cs_tsinstr_yrobust"],
        help=(
            "Scaling ids include: raw_yraw raw_tsall_yz ts_raw_yz cs_none_yraw "
            "cs_tsinstr_yz cs_tsmkt_yz cs_tsall_yz cs_tsall_yraw cs_tsinstr_yrobust"
        ),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["baseline", "gex", "flow", "liquidity", "all"],
        help="Feature set ids: baseline gex flow liquidity all",
    )
    parser.add_argument(
        "--representative-moneyness",
        action="store_true",
        default=True,
        help=(
            "Use one representative moneyness configuration per strategy in conditional tests "
            "(default: on)."
        ),
    )
    parser.add_argument(
        "--all-moneyness",
        action="store_false",
        dest="representative_moneyness",
        help="Disable representative-moneyness filter and use all strategy moneyness configurations.",
    )
    parser.add_argument(
        "--max-moneyness-dev",
        type=float,
        default=0.01,
        help=(
            "Maximum absolute deviation from 1.0 across legs for candidate representative "
            "moneyness configurations."
        ),
    )
    parser.add_argument(
        "--net-cost",
        type=float,
        default=0.005,
        help="Per-trade cost in reth_und units, subtracted from realized return.",
    )
    parser.add_argument(
        "--nn-max-epochs",
        type=int,
        default=120,
        help="Maximum NN training epochs for PyTorch model variants.",
    )
    parser.add_argument(
        "--nn-patience",
        type=int,
        default=14,
        help="Early-stopping patience for PyTorch model variants.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Summary CSV output path. Default: <root>/data/conditional_model_zoo_summary.csv",
    )
    parser.add_argument(
        "--pred-out",
        type=Path,
        default=None,
        help="Prediction parquet output path. Default: <root>/data/conditional_model_zoo_predictions.parquet",
    )
    parser.add_argument(
        "--store-preds",
        action="store_true",
        default=True,
        help="Store OOS predictions parquet (default: on).",
    )
    parser.add_argument(
        "--no-store-preds",
        action="store_false",
        dest="store_preds",
        help="Disable storing OOS prediction rows; useful for large model grids.",
    )
    parser.add_argument(
        "--rep-moneyness-out",
        type=Path,
        default=None,
        help="Representative strategy-moneyness CSV path. Default: <root>/data/conditional_representative_moneyness.csv",
    )
    parser.add_argument(
        "--latex-out",
        type=Path,
        default=None,
        help="LaTeX table output path. Default: <root>/output/tables/0dte_conditional_model_zoo.tex",
    )
    parser.add_argument(
        "--latex-compact-out",
        type=Path,
        default=None,
        help="Compact LaTeX table output path. Default: <root>/output/tables/0dte_conditional_model_zoo_compact.tex",
    )
    parser.add_argument(
        "--latex-tree-compact-out",
        type=Path,
        default=None,
        help="Tree-horserace compact LaTeX output path.",
    )
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def cs_zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if pd.isna(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd


def _build_leg_feature_frame(strats: pd.DataFrame, opt: pd.DataFrame) -> pd.DataFrame:
    """Construct strategy-level liquidity/flow/GEX features from option legs."""
    if strats.empty or opt.empty:
        return pd.DataFrame(columns=["row_id"])

    if "mnes" not in opt.columns:
        return pd.DataFrame(columns=["row_id"])

    opt = opt.copy()
    opt["quote_date"] = pd.to_datetime(opt["quote_date"])
    opt["quote_time"] = opt["quote_time"].astype(str)
    opt["mnes_int"] = pd.to_numeric(opt["mnes"], errors="coerce").round().astype("Int64")
    opt = opt.dropna(subset=["mnes_int"]).copy()
    opt["mnes_int"] = opt["mnes_int"].astype(int)

    fill_zero_cols = [
        "trade_volume",
        "open_interest",
        "bid_size",
        "ask_size",
    ]
    for col in fill_zero_cols:
        if col not in opt.columns:
            opt[col] = 0.0
        opt[col] = pd.to_numeric(opt[col], errors="coerce").fillna(0.0)

    core_cols = [
        "bas",
        "mid",
        "delta",
        "gamma",
        "vega",
        "active_underlying_price",
        "trade_volume",
        "open_interest",
        "bid_size",
        "ask_size",
    ]
    for col in core_cols:
        if col not in opt.columns:
            opt[col] = np.nan
        opt[col] = pd.to_numeric(opt[col], errors="coerce")

    agg_map: dict[str, str] = {
        "bas": "mean",
        "mid": "mean",
        "delta": "mean",
        "gamma": "mean",
        "vega": "mean",
        "active_underlying_price": "mean",
        "trade_volume": "sum",
        "open_interest": "mean",
        "bid_size": "sum",
        "ask_size": "sum",
    }
    lookup = (
        opt.groupby(["quote_date", "quote_time", "option_type", "mnes_int"], as_index=False)
        .agg(agg_map)
        .rename(columns={"option_type": "leg_option_type"})
    )

    legs_records: list[dict[str, object]] = []
    for row in strats[["row_id", "quote_date", "quote_time", "option_type", "mnes"]].itertuples(index=False):
        legs = get_legs(strategy=str(row.option_type), mnes_str=str(row.mnes))
        for leg_idx, (leg_opt_type, leg_mnes, qty) in enumerate(legs):
            legs_records.append(
                {
                    "row_id": int(row.row_id),
                    "quote_date": pd.to_datetime(row.quote_date),
                    "quote_time": str(row.quote_time),
                    "leg_option_type": str(leg_opt_type),
                    "mnes_int": int(leg_mnes),
                    "qty": float(qty),
                    "leg_idx": int(leg_idx),
                }
            )

    if not legs_records:
        return pd.DataFrame({"row_id": strats["row_id"].to_numpy()})

    legs_df = pd.DataFrame(legs_records)
    legs_df = legs_df.merge(
        lookup,
        how="left",
        on=["quote_date", "quote_time", "leg_option_type", "mnes_int"],
    )
    legs_df["leg_found"] = (~legs_df["mid"].isna()).astype(float)

    work = legs_df.copy()
    for col in core_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    qty = work["qty"].astype(float)
    qty_abs = qty.abs()
    s = work["active_underlying_price"].astype(float).clip(lower=0.0)

    work["liq_halfspread"] = qty_abs * work["bas"].clip(lower=0.0) / 2.0
    work["liq_depth"] = qty_abs * (work["bid_size"].clip(lower=0.0) + work["ask_size"].clip(lower=0.0))
    work["liq_rel_spread"] = qty_abs * (
        work["bas"].clip(lower=0.0) / (work["mid"].abs().replace(0.0, np.nan))
    )
    work["flow_trade_volume"] = qty_abs * work["trade_volume"].clip(lower=0.0)
    work["flow_trade_delta_usd"] = qty_abs * work["trade_volume"].clip(lower=0.0) * work["delta"].abs() * 100.0 * s
    work["flow_trade_gamma_usd"] = (
        qty_abs * work["trade_volume"].clip(lower=0.0) * work["gamma"].abs() * 100.0 * (s ** 2)
    )
    work["flow_trade_vega_usd"] = qty_abs * work["trade_volume"].clip(lower=0.0) * work["vega"].abs() * 100.0
    work["gex_oi_gamma_net_usd"] = qty * work["open_interest"].clip(lower=0.0) * work["gamma"] * 100.0 * (s ** 2)
    work["gex_oi_gamma_abs_usd"] = qty_abs * work["open_interest"].clip(lower=0.0) * work["gamma"].abs() * 100.0 * (s ** 2)
    work["gex_gamma_net"] = qty * work["gamma"]
    work["gex_gamma_abs"] = qty_abs * work["gamma"].abs()

    agg_cols = [
        "liq_halfspread",
        "liq_depth",
        "liq_rel_spread",
        "flow_trade_volume",
        "flow_trade_delta_usd",
        "flow_trade_gamma_usd",
        "flow_trade_vega_usd",
        "gex_oi_gamma_net_usd",
        "gex_oi_gamma_abs_usd",
        "gex_gamma_net",
        "gex_gamma_abs",
    ]
    out = work.groupby("row_id", as_index=False)[agg_cols].sum()
    coverage = work.groupby("row_id", as_index=False)["leg_found"].mean().rename(columns={"leg_found": "leg_coverage"})
    out = out.merge(coverage, how="left", on="row_id")

    out["gex_balance"] = out["gex_oi_gamma_net_usd"] / (out["gex_oi_gamma_abs_usd"].abs() + 1.0)
    out["flow_gamma_to_oi"] = out["flow_trade_gamma_usd"] / (out["gex_oi_gamma_abs_usd"].abs() + 1.0)
    out["liq_spread_to_depth"] = out["liq_halfspread"] / (out["liq_depth"] + 1.0)

    feature_cols = [c for c in out.columns if c not in {"row_id", "leg_coverage"}]
    out.loc[out["leg_coverage"] < 0.999, feature_cols] = np.nan
    return out


def prepare_dataset(
    data_dir: Path,
    *,
    representative_moneyness: bool,
    max_moneyness_dev: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strats = pd.read_parquet(data_dir / "data_structures.parquet")
    opt = pd.read_parquet(data_dir / "data_opt.parquet")
    vix = pd.read_parquet(data_dir / "vix.parquet")
    slopes = pd.read_parquet(data_dir / "slopes.parquet")
    ex_post_file = data_dir / "ex_post_moments.h5"

    strats = strats.copy()
    strats["quote_date"] = pd.to_datetime(strats["quote_date"])
    strats["quote_time"] = strats["quote_time"].astype(str)
    strats = strats[strats["quote_time"] == "10:00:00"].copy()

    opt = opt.copy()
    opt["quote_date"] = pd.to_datetime(opt["quote_date"])
    opt["quote_time"] = opt["quote_time"].astype(str)
    opt = opt[opt["quote_time"] == "10:00:00"].copy()

    exclude = {"C", "P", "S"}
    strats = strats[~strats["option_type"].astype(str).isin(exclude)].copy()

    selected_mnes = pd.DataFrame(columns=["option_type", "mnes", "rows", "days", "max_moneyness_dev"])
    if representative_moneyness:
        selected_mnes = choose_representative_moneyness(
            strats[["quote_date", "option_type", "mnes"]],
            cfg=RepresentativeSelectionConfig(max_moneyness_dev=float(max_moneyness_dev)),
        )
        if selected_mnes.empty:
            raise RuntimeError(
                f"No representative moneyness configurations found with max deviation <= {max_moneyness_dev:.4f}."
            )
        strats = apply_representative_filter(strats, selected_mnes)

    strats = strats.reset_index(drop=True)
    strats["row_id"] = np.arange(len(strats), dtype=int)

    def mnes_center(v: str) -> float:
        vals = [float(x) for x in str(v).split("/")]
        return float(np.mean(vals))

    strats["mnes_center"] = strats["mnes"].astype(str).map(mnes_center)
    leg_feats = _build_leg_feature_frame(strats=strats, opt=opt)
    strats = strats.merge(leg_feats, how="left", on="row_id")

    inst_cols = ["option_type", "mnes"]
    strats = strats.sort_values(inst_cols + ["quote_date"])
    grp = strats.groupby(inst_cols, observed=True)
    strats["pnl_l1"] = grp["reth_und"].shift(1)
    strats["pnl_mean5_l1"] = grp["pnl_l1"].transform(lambda s: s.rolling(5).mean())
    strats["pnl_std5_l1"] = grp["pnl_l1"].transform(lambda s: s.rolling(5).std())

    feat = build_feature_frame(vix=vix, slopes=slopes, ex_post_file=ex_post_file)
    data = strats.merge(feat, how="inner", on="quote_date")

    instr_base = ["mid", "delta", "gamma", "vega", "mnes_center", "pnl_l1", "pnl_mean5_l1", "pnl_std5_l1"]
    if "tv" in data.columns:
        instr_base.insert(1, "tv")

    instr_gex = ["gex_oi_gamma_net_usd", "gex_oi_gamma_abs_usd", "gex_balance", "gex_gamma_net", "gex_gamma_abs"]
    instr_flow = [
        "flow_trade_volume",
        "flow_trade_delta_usd",
        "flow_trade_gamma_usd",
        "flow_trade_vega_usd",
        "flow_gamma_to_oi",
    ]
    instr_liq = ["liq_halfspread", "liq_depth", "liq_rel_spread", "liq_spread_to_depth"]

    market_cols = ["iv", "isk", "slope_up", "slope_dn", "SPX_lret", "SPX_lrv", "SPX_lrv_skew"]

    instr_all = [
        c for c in (instr_base + instr_gex + instr_flow + instr_liq)
        if c in data.columns
    ]
    cs_cols = instr_all
    for col in cs_cols:
        data[f"{col}_cs"] = data.groupby("quote_date", observed=True)[col].transform(cs_zscore)

    dummies = pd.get_dummies(data["option_type"].astype(str), prefix="strat", dtype=float)
    data = pd.concat([data, dummies], axis=1)

    dummy_cols = list(dummies.columns)

    data["y"] = data["reth_und"].astype(float)
    data["y_bin"] = (data["y"] > 0.0).astype(int)

    data.attrs["market_raw"] = [c for c in market_cols if c in data.columns]
    data.attrs["dummy_cols"] = dummy_cols
    data.attrs["instr_groups_raw"] = {
        "baseline": [c for c in instr_base if c in data.columns],
        "gex": [c for c in instr_gex if c in data.columns],
        "flow": [c for c in instr_flow if c in data.columns],
        "liquidity": [c for c in instr_liq if c in data.columns],
    }
    data.attrs["instr_groups_cs"] = {
        k: [f"{c}_cs" for c in vals if f"{c}_cs" in data.columns]
        for k, vals in data.attrs["instr_groups_raw"].items()
    }

    return data, selected_mnes


def make_scaling_specs() -> dict[str, ScalingSpec]:
    return {
        "ts_raw_yz": ScalingSpec(name="ts_raw_yz", feature_space="raw", ts_mode="all", y_mode="zscore"),
        "cs_ts_yz": ScalingSpec(name="cs_ts_yz", feature_space="cs_mix", ts_mode="all", y_mode="zscore"),
        "cs_ts_yraw": ScalingSpec(name="cs_ts_yraw", feature_space="cs_mix", ts_mode="all", y_mode="raw"),
        "raw_yraw": ScalingSpec(name="raw_yraw", feature_space="raw", ts_mode="none", y_mode="raw"),
        "raw_tsall_yz": ScalingSpec(name="raw_tsall_yz", feature_space="raw", ts_mode="all", y_mode="zscore"),
        "cs_none_yraw": ScalingSpec(name="cs_none_yraw", feature_space="cs_mix", ts_mode="none", y_mode="raw"),
        "cs_tsinstr_yz": ScalingSpec(name="cs_tsinstr_yz", feature_space="cs_mix", ts_mode="instr_only", y_mode="zscore"),
        "cs_tsmkt_yz": ScalingSpec(name="cs_tsmkt_yz", feature_space="cs_mix", ts_mode="market_only", y_mode="zscore"),
        "cs_tsall_yz": ScalingSpec(name="cs_tsall_yz", feature_space="cs_mix", ts_mode="all", y_mode="zscore"),
        "cs_tsall_yraw": ScalingSpec(name="cs_tsall_yraw", feature_space="cs_mix", ts_mode="all", y_mode="raw"),
        "cs_tsinstr_yrobust": ScalingSpec(
            name="cs_tsinstr_yrobust",
            feature_space="cs_mix",
            ts_mode="instr_only",
            y_mode="robust",
        ),
    }


def default_model_ids(task: str) -> list[str]:
    if task == "binary":
        return [
            "logit",
            "ridge_logit",
            "elastic_net_logit",
            "rf_clf",
            "et_clf",
            "hgb_clf",
            "lgbm_clf",
            "xgb_clf",
            "catboost_clf",
            "nn_linear_ridge_clf",
            "nn_cnn_clf",
            "nn_relu_clf",
            "nn_tanh_clf",
        ]

    return [
        "ridge",
        "elastic_net",
        "lgbm",
        "rf",
        "xgb",
        "catboost",
        "nn_linear_ridge",
        "nn_cnn",
        "nn_relu",
        "nn_tanh",
        "nn_logistic",
    ]


def make_model_zoo(
    task: str = "regression",
    random_state: int = 0,
    nn_max_epochs: int = 120,
    nn_patience: int = 14,
) -> dict[str, object]:
    nn_max_epochs = int(max(5, nn_max_epochs))
    nn_patience = int(max(2, min(nn_patience, nn_max_epochs - 1)))
    ridge_alphas = np.logspace(-4, 4, 25)
    ridge_alphas_strong = np.logspace(-2, 6, 25)
    ridge_cv = TimeSeriesSplit(n_splits=5)

    if task == "binary":
        zoo: dict[str, object] = {
            "logit": LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=5000,
                random_state=random_state,
            ),
            "ridge_logit": LogisticRegression(
                C=0.2,
                solver="lbfgs",
                max_iter=5000,
                random_state=random_state,
            ),
            "elastic_net_logit": LogisticRegression(
                l1_ratio=0.5,
                C=0.35,
                solver="saga",
                max_iter=7000,
                random_state=random_state,
            ),
            "rf_clf": RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=50,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=4,
            ),
            "et_clf": ExtraTreesClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=50,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=4,
            ),
            "hgb_clf": HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_depth=5,
                max_leaf_nodes=31,
                min_samples_leaf=60,
                l2_regularization=0.3,
                max_iter=260,
                random_state=random_state,
            ),
            "nn_linear_ridge_clf": TorchBinaryClassifier(
                model_type="linear_ridge",
                init_alpha=2.0,
                min_alpha=1e-7,
                alpha_prior_strength=5e-4,
                lr=1e-3,
                weight_decay=1e-5,
                max_epochs=nn_max_epochs,
                patience=nn_patience,
                random_state=random_state,
            ),
            "nn_cnn_clf": TorchBinaryClassifier(
                model_type="cnn",
                channels=(32, 16),
                kernel_size=3,
                activation="relu",
                dropout=0.05,
                lr=9e-4,
                weight_decay=1e-4,
                max_epochs=nn_max_epochs,
                patience=nn_patience,
                random_state=random_state,
            ),
            "nn_relu_clf": TorchBinaryClassifier(
                model_type="mlp",
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                dropout=0.05,
                lr=1e-3,
                weight_decay=1e-4,
                max_epochs=nn_max_epochs,
                patience=nn_patience,
                random_state=random_state,
            ),
            "nn_tanh_clf": TorchBinaryClassifier(
                model_type="mlp",
                hidden_layer_sizes=(256, 128, 64),
                activation="tanh",
                dropout=0.04,
                lr=8e-4,
                weight_decay=8e-5,
                max_epochs=nn_max_epochs,
                patience=nn_patience,
                random_state=random_state,
            ),
        }

        if HAS_LIGHTGBM:
            zoo["lgbm_clf"] = LGBMClassifier(
                n_estimators=180,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                objective="binary",
                verbose=-1,
            )
            zoo["lgbm_clf_deep"] = LGBMClassifier(
                n_estimators=320,
                learning_rate=0.02,
                num_leaves=63,
                max_depth=8,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.8,
                random_state=random_state,
                objective="binary",
                verbose=-1,
            )
        if HAS_XGBOOST:
            zoo["xgb_clf"] = XGBClassifier(
                n_estimators=260,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=1.0,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=4,
                verbosity=0,
            )
            zoo["xgb_clf_deep"] = XGBClassifier(
                n_estimators=460,
                learning_rate=0.02,
                max_depth=6,
                min_child_weight=3.0,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=4,
                verbosity=0,
            )
        if HAS_CATBOOST:
            zoo["catboost_clf"] = CatBoostClassifier(
                loss_function="Logloss",
                iterations=320,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            )
            zoo["catboost_clf_deep"] = CatBoostClassifier(
                loss_function="Logloss",
                iterations=540,
                learning_rate=0.02,
                depth=8,
                l2_leaf_reg=4.0,
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            )
        return zoo

    zoo = {
        "ridge": RidgeCV(alphas=ridge_alphas, cv=ridge_cv),
        "ridge_strong": RidgeCV(alphas=ridge_alphas_strong, cv=ridge_cv),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.35, max_iter=5000, random_state=random_state),
        "elastic_net_sparse": ElasticNet(alpha=0.02, l1_ratio=0.7, max_iter=7000, random_state=random_state),
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=4,
        ),
        "rf_deep": RandomForestRegressor(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=4,
        ),
        "nn_linear_ridge": TorchLinearRidgeRegressor(
            init_alpha=2.0,
            min_alpha=1e-7,
            alpha_prior_strength=5e-4,
            lr=1e-3,
            max_epochs=nn_max_epochs,
            patience=nn_patience,
            random_state=random_state,
        ),
        "nn_cnn": TorchCNNRegressor(
            channels=(32, 16),
            kernel_size=3,
            activation="relu",
            dropout=0.04,
            lr=9e-4,
            weight_decay=1e-4,
            max_epochs=nn_max_epochs,
            patience=nn_patience,
            random_state=random_state,
        ),
        "nn_relu": TorchMLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            dropout=0.05,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=nn_max_epochs,
            patience=nn_patience,
            random_state=random_state,
        ),
        "nn_relu_deep": TorchMLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation="relu",
            dropout=0.08,
            lr=8e-4,
            weight_decay=1e-4,
            max_epochs=nn_max_epochs,
            patience=nn_patience,
            random_state=random_state,
        ),
        "nn_tanh": TorchMLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="tanh",
            dropout=0.03,
            lr=7e-4,
            weight_decay=5e-5,
            max_epochs=nn_max_epochs,
            patience=nn_patience,
            random_state=random_state,
        ),
        "nn_logistic": TorchMLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="logistic",
            dropout=0.0,
            lr=7e-4,
            weight_decay=8e-5,
            max_epochs=nn_max_epochs,
            patience=nn_patience,
            random_state=random_state,
        ),
    }

    if HAS_LIGHTGBM:
        zoo["lgbm"] = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            objective="regression",
            verbose=-1,
        )
        zoo["lgbm_deep"] = LGBMRegressor(
            n_estimators=260,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=8,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="regression",
            verbose=-1,
        )
    if HAS_XGBOOST:
        zoo["xgb"] = XGBRegressor(
            n_estimators=220,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=1.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=4,
            verbosity=0,
        )
        zoo["xgb_deep"] = XGBRegressor(
            n_estimators=420,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=3.0,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=4,
            verbosity=0,
        )
    if HAS_CATBOOST:
        zoo["catboost"] = CatBoostRegressor(
            loss_function="RMSE",
            iterations=260,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
        zoo["catboost_deep"] = CatBoostRegressor(
            loss_function="RMSE",
            iterations=480,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=4.0,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
    return zoo


def select_features(data: pd.DataFrame, spec: ScalingSpec, feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    if spec.feature_space == "cs_mix":
        groups = data.attrs["instr_groups_cs"]
    elif spec.feature_space == "raw":
        groups = data.attrs["instr_groups_raw"]
    else:
        raise ValueError(f"Unsupported feature_space: {spec.feature_space}")

    instr_cols = list(groups.get("baseline", []))
    if feature_set in {"gex", "all"}:
        instr_cols += list(groups.get("gex", []))
    if feature_set in {"flow", "all"}:
        instr_cols += list(groups.get("flow", []))
    if feature_set in {"liquidity", "all"}:
        instr_cols += list(groups.get("liquidity", []))

    instr_cols = list(dict.fromkeys(instr_cols))

    market_cols = list(data.attrs["market_raw"])
    dummy_cols = list(data.attrs.get("dummy_cols", []))
    features = [c for c in (market_cols + instr_cols + dummy_cols) if c in data.columns]

    if spec.ts_mode == "none":
        ts_cols = []
    elif spec.ts_mode == "all":
        ts_cols = market_cols + instr_cols
    elif spec.ts_mode == "market_only":
        ts_cols = market_cols
    elif spec.ts_mode == "instr_only":
        ts_cols = instr_cols
    else:
        raise ValueError(f"Unsupported ts_mode: {spec.ts_mode}")

    return features, ts_cols


def fit_y_transform(y_train: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, float] | None]:
    if mode == "raw":
        return y_train, None

    if mode == "zscore":
        mu = float(np.nanmean(y_train))
        sd = float(np.nanstd(y_train))
        if (not np.isfinite(sd)) or sd <= 1e-12:
            sd = 1.0
        return (y_train - mu) / sd, {"kind": "zscore", "mu": mu, "scale": sd}

    if mode == "robust":
        med = float(np.nanmedian(y_train))
        mad = float(np.nanmedian(np.abs(y_train - med)))
        scale = mad * 1.4826
        if (not np.isfinite(scale)) or scale <= 1e-12:
            scale = float(np.nanstd(y_train))
        if (not np.isfinite(scale)) or scale <= 1e-12:
            scale = 1.0
        return (y_train - med) / scale, {"kind": "robust", "mu": med, "scale": scale}

    raise ValueError(f"Unsupported y transform: {mode}")


def inv_y_transform(y_pred: np.ndarray, y_map: dict[str, float] | None) -> np.ndarray:
    if y_map is None:
        return y_pred
    return y_pred * float(y_map["scale"]) + float(y_map["mu"])


def prepare_work(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    work = data.dropna(subset=features + ["y", "y_bin"]).copy()
    work = work.sort_values(["quote_date", "option_type", "mnes"]).reset_index(drop=True)
    return work


def _is_tree_dataframe_model(model: object) -> bool:
    return model.__class__.__name__.startswith(("LGBM", "XGB", "CatBoost"))


def _predict_binary_probabilities(model, X_input) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X_input), dtype=float)
        if p.ndim == 2 and p.shape[1] >= 2:
            p1 = p[:, 1]
        else:
            p1 = p.reshape(-1)
    elif hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X_input), dtype=float).reshape(-1)
        p1 = sigmoid(score)
    else:
        raw = np.asarray(model.predict(X_input), dtype=float).reshape(-1)
        if raw.size == 0:
            p1 = raw
        elif float(np.nanmin(raw)) >= 0.0 and float(np.nanmax(raw)) <= 1.0:
            p1 = raw
        else:
            p1 = sigmoid(raw)
    return np.clip(np.asarray(p1, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)


def _positions_from_probability(
    p_hat: np.ndarray,
    *,
    threshold: float,
    decision_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(p_hat, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
    th = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    sign = np.where(p >= th, 1.0, -1.0)
    if decision_mode == "hard":
        return sign, sign
    if decision_mode == "soft":
        up = max(1.0 - th, 1e-6)
        dn = max(th, 1e-6)
        w = np.where(p >= th, (p - th) / up, -((th - p) / dn))
        return sign, np.clip(w, -1.0, 1.0)
    raise ValueError(f"Unsupported decision_mode: {decision_mode}")


def walk_forward_predict(
    work: pd.DataFrame,
    features: list[str],
    ts_scale_cols: list[str],
    model,
    protocol: str,
    min_train_days: int,
    rolling_window: int,
    refit_every: int,
    y_mode: str,
    task: str,
    decision_mode: str,
    decision_threshold: float,
) -> pd.DataFrame:
    if work.empty:
        return pd.DataFrame()

    date_grp = work.groupby("quote_date", sort=True, observed=True).size()
    dates = date_grp.index.to_numpy()
    counts = date_grp.to_numpy(dtype=int)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    stops = np.cumsum(counts)

    if len(dates) <= min_train_days:
        return pd.DataFrame()

    X_all = work[features].to_numpy(dtype=float)
    y_all = work["y"].to_numpy(dtype=float)
    y_bin_all = work["y_bin"].to_numpy(dtype=int)
    d_all = work["quote_date"].to_numpy()
    opt_all = work["option_type"].astype(str).to_numpy()
    mnes_all = work["mnes"].astype(str).to_numpy()

    ts_set = set(ts_scale_cols)
    ts_idx = [i for i, col in enumerate(features) if col in ts_set]

    preds: list[pd.DataFrame] = []
    fitted = None
    x_scaler = None
    y_map: dict[str, float] | None = None
    constant_prob: float | None = None
    tree_dataframe_model = _is_tree_dataframe_model(model)

    for i in range(min_train_days, len(dates)):
        if protocol == "expanding":
            train_day_start = 0
        elif protocol == "rolling":
            train_day_start = max(0, i - rolling_window)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        tr_start = int(starts[train_day_start])
        tr_stop = int(starts[i])
        te_start = int(starts[i])
        te_stop = int(stops[i])
        if tr_stop <= tr_start or te_stop <= te_start:
            continue

        X_tr = X_all[tr_start:tr_stop].copy()
        X_te = X_all[te_start:te_stop].copy()
        y_tr = y_all[tr_start:tr_stop]
        y_tr_bin = y_bin_all[tr_start:tr_stop]

        do_refit = (fitted is None) or (((i - min_train_days) % max(refit_every, 1)) == 0)

        if do_refit:
            if ts_idx:
                x_scaler = StandardScaler()
                X_tr[:, ts_idx] = x_scaler.fit_transform(X_tr[:, ts_idx])
                X_te[:, ts_idx] = x_scaler.transform(X_te[:, ts_idx])
            else:
                x_scaler = None

            try:
                fitted = clone(model)
                if task == "binary":
                    if np.unique(y_tr_bin).size < 2:
                        fitted = None
                        constant_prob = float(np.clip(np.mean(y_tr_bin), 1e-6, 1.0 - 1e-6))
                    else:
                        constant_prob = None
                        y_fit_bin = y_tr_bin.astype(int)
                        if tree_dataframe_model:
                            fitted.fit(pd.DataFrame(X_tr, columns=features), y_fit_bin)
                        else:
                            fitted.fit(X_tr, y_fit_bin)
                else:
                    y_fit, y_map = fit_y_transform(y_tr, mode=y_mode)
                    constant_prob = None
                    if tree_dataframe_model:
                        fitted.fit(pd.DataFrame(X_tr, columns=features), y_fit)
                    else:
                        fitted.fit(X_tr, y_fit)
            except Exception as exc:
                warnings.warn(
                    (
                        f"Model fit failed for protocol={protocol}, date={pd.Timestamp(dates[i]).date()}, "
                        f"task={task}. Falling back to constant prediction. Error: {exc}"
                    ),
                    RuntimeWarning,
                )
                if task == "binary":
                    fitted = None
                    constant_prob = float(np.clip(np.mean(y_tr_bin), 1e-6, 1.0 - 1e-6))
                else:
                    fitted = None
                    constant_prob = None
                    continue
        else:
            if x_scaler is not None and ts_idx:
                X_te[:, ts_idx] = x_scaler.transform(X_te[:, ts_idx])

        X_te_in = pd.DataFrame(X_te, columns=features) if tree_dataframe_model else X_te

        if task == "binary":
            if constant_prob is not None:
                p_hat = np.full(te_stop - te_start, float(constant_prob), dtype=float)
            elif fitted is not None:
                p_hat = _predict_binary_probabilities(fitted, X_te_in)
            else:
                p_hat = np.full(te_stop - te_start, 0.5, dtype=float)
            yhat = p_hat.copy()
            sign, weight = _positions_from_probability(
                p_hat,
                threshold=float(decision_threshold),
                decision_mode=decision_mode,
            )
        else:
            if fitted is None:
                continue
            if tree_dataframe_model:
                yhat = fitted.predict(X_te_in).astype(float)
            else:
                yhat = fitted.predict(X_te).astype(float)
            yhat = inv_y_transform(yhat, y_map=y_map)

            train_std = float(np.nanstd(y_tr))
            if not np.isfinite(train_std) or train_std <= 1e-10:
                train_std = 1.0

            p_hat = np.clip(sigmoid(yhat / train_std), 1e-6, 1.0 - 1e-6)
            sign, weight = _positions_from_probability(
                p_hat,
                threshold=float(decision_threshold),
                decision_mode=decision_mode,
            )

        out = pd.DataFrame(
            {
                "quote_date": d_all[te_start:te_stop],
                "option_type": opt_all[te_start:te_stop],
                "mnes": mnes_all[te_start:te_stop],
                "y": y_all[te_start:te_stop],
                "y_bin": y_bin_all[te_start:te_stop],
                "yhat": yhat,
                "p_hat": p_hat,
                "sign": sign,
                "weight": weight,
            }
        )
        preds.append(out)

    if not preds:
        return pd.DataFrame()
    return pd.concat(preds, axis=0, ignore_index=True)


def summarize_predictions(pred: pd.DataFrame, net_cost: float) -> dict[str, float]:
    if pred.empty:
        return {}

    y_true = pred["y"].to_numpy(dtype=float)
    y_bin = pred["y_bin"].to_numpy(dtype=float)
    sign_true = np.where(y_true >= 0.0, 1.0, -1.0)
    sign_pred = pred["sign"].to_numpy(dtype=float)
    p_hat = np.clip(pred["p_hat"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    if "weight" in pred.columns:
        weights = pred["weight"].to_numpy(dtype=float)
    else:
        weights = sign_pred.copy()

    hit_rate = float(np.mean(sign_pred == sign_true))
    brier = float(np.mean((p_hat - y_bin) ** 2))
    calib = float(calibr_slope(y_true=y_bin, p_hat=p_hat))
    try:
        logloss_val = float(log_loss(y_true=y_bin.astype(int), y_pred=p_hat, labels=[0, 1]))
    except Exception:
        logloss_val = np.nan
    if np.unique(y_bin).size >= 2:
        auc = float(roc_auc_score(y_true=y_bin, y_score=p_hat))
    else:
        auc = np.nan

    dir_pnl_net = weights * (y_true - float(net_cost))
    pred = pred.copy()
    pred["dir_pnl_net"] = dir_pnl_net

    daily = pred.groupby("quote_date", observed=True)["dir_pnl_net"].mean().sort_index()
    sr_net = float(annualized_sharpe(daily))

    return {
        "hit_rate": hit_rate,
        "brier": brier,
        "calib_slope": calib,
        "logloss": logloss_val,
        "auc": auc,
        "long_share": float(np.mean(sign_pred > 0)),
        "avg_abs_weight": float(np.mean(np.abs(weights))),
        "mean_net_bp": float(np.mean(dir_pnl_net) * 100.0),
        "sr_net": sr_net,
        "obs": int(pred.shape[0]),
        "days": int(daily.shape[0]),
    }


def fmt(x: float | int | None, n: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def tex_escape(v: object) -> str:
    s = str(v)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, vv in repl.items():
        s = s.replace(k, vv)
    return s


MODEL_LABELS = {
    "logit": "Logistic",
    "ridge_logit": "Logistic (Ridge)",
    "elastic_net_logit": "Logistic (Elastic Net)",
    "ridge": "Ridge (CV)",
    "ridge_strong": "Ridge (CV Strong)",
    "elastic_net": "Elastic Net",
    "elastic_net_sparse": "Elastic Net (Sparse)",
    "lgbm": "LightGBM",
    "lgbm_deep": "LightGBM (Deep)",
    "lgbm_clf": "LightGBM Classifier",
    "lgbm_clf_deep": "LightGBM Classifier (Deep)",
    "rf": "Random Forest",
    "rf_deep": "Random Forest (Deep)",
    "rf_clf": "Random Forest Classifier",
    "et_clf": "Extra Trees Classifier",
    "hgb_clf": "HistGB Classifier",
    "xgb": "XGBoost",
    "xgb_deep": "XGBoost (Deep)",
    "xgb_clf": "XGBoost Classifier",
    "xgb_clf_deep": "XGBoost Classifier (Deep)",
    "catboost": "CatBoost",
    "catboost_deep": "CatBoost (Deep)",
    "catboost_clf": "CatBoost Classifier",
    "catboost_clf_deep": "CatBoost Classifier (Deep)",
    "nn_linear_ridge": "Neural Net (Linear + Trainable Ridge)",
    "nn_linear_ridge_clf": "Neural Net (Linear + Trainable Ridge, Clf)",
    "nn_cnn": "Neural Net (CNN)",
    "nn_cnn_clf": "Neural Net (CNN, Clf)",
    "nn_relu": "Neural Net (ReLU)",
    "nn_relu_deep": "Neural Net (ReLU Deep)",
    "nn_relu_clf": "Neural Net (ReLU, Clf)",
    "nn_tanh": "Neural Net (Tanh)",
    "nn_tanh_clf": "Neural Net (Tanh, Clf)",
    "nn_logistic": "Neural Net (Logistic)",
}

FEATURE_SET_LABELS = {
    "baseline": "Baseline",
    "gex": "Gamma Exposure",
    "flow": "Flow",
    "liquidity": "Liquidity",
    "all": "All Features",
}

PROTOCOL_LABELS = {
    "expanding": "Expanding",
    "rolling": "Rolling",
}

SCALING_LABELS = {
    "ts_raw_yz": "Raw features + time-series scaling (all), response z-score",
    "cs_ts_yz": "Cross-sectional + time-series scaling (all), response z-score",
    "cs_ts_yraw": "Cross-sectional + time-series scaling (all), response raw",
    "raw_yraw": "Raw features, response raw",
    "raw_tsall_yz": "Raw features + time-series scaling (all), response z-score",
    "cs_none_yraw": "Cross-sectional only, response raw",
    "cs_tsinstr_yz": "Cross-sectional + time-series scaling (strategy features), response z-score",
    "cs_tsmkt_yz": "Cross-sectional + time-series scaling (market features), response z-score",
    "cs_tsall_yz": "Cross-sectional + time-series scaling (all), response z-score",
    "cs_tsall_yraw": "Cross-sectional + time-series scaling (all), response raw",
    "cs_tsinstr_yrobust": "Cross-sectional + time-series scaling (strategy features), response robust z-score",
}


def display_label(value: object, mapping: dict[str, str]) -> str:
    key = str(value)
    return mapping.get(key, key)


def write_latex(summary: pd.DataFrame, output_file: Path) -> None:
    lines = [
        r"\begin{tabular}{llllrrrrrr}",
        r"\toprule",
        r"Model & Feature Set & Scaling & Window & Hit Rate (\%) & Brier & Calib. Slope & Mean Net (bps) & SR Net & N \\",
        r"\midrule",
    ]

    for row in summary.itertuples(index=False):
        model = display_label(row.model, MODEL_LABELS)
        feature_set = display_label(row.feature_set, FEATURE_SET_LABELS)
        scaling = display_label(row.scaling, SCALING_LABELS)
        protocol = display_label(row.protocol, PROTOCOL_LABELS)
        lines.append(
            f"{tex_escape(model)} & {tex_escape(feature_set)} & {tex_escape(scaling)} & {tex_escape(protocol)} & {fmt(row.hit_rate * 100.0, 1)} & "
            f"{fmt(row.brier, 3)} & {fmt(row.calib_slope, 2)} & {fmt(row.mean_net_bp, 3)} & "
            f"{fmt(row.sr_net, 2)} & {int(row.obs)} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_compact(summary: pd.DataFrame, output_file: Path) -> None:
    best = (
        summary.sort_values(["mean_net_bp", "sr_net"], ascending=[False, False])
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best = best.sort_values(["mean_net_bp", "sr_net"], ascending=[False, False]).reset_index(drop=True)
    top = best.iloc[0] if not best.empty else None

    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule", r"Model & Best Feature Set & Window & Hit Rate (\%) & Mean Net (bps) & SR Net & N \\", r"\midrule"]
    for row in best.itertuples(index=False):
        model = display_label(row.model, MODEL_LABELS)
        feature_set = display_label(row.feature_set, FEATURE_SET_LABELS)
        protocol = display_label(row.protocol, PROTOCOL_LABELS)
        lines.append(
            f"{tex_escape(model)} & {tex_escape(feature_set)} & {tex_escape(protocol)} & "
            f"{fmt(row.hit_rate * 100.0, 1)} & {fmt(row.mean_net_bp, 3)} & {fmt(row.sr_net, 2)} & {int(row.obs)} \\\\"
        )

    lines.append(r"\midrule")
    if top is not None:
        feature_set = display_label(top.feature_set, FEATURE_SET_LABELS)
        protocol = display_label(top.protocol, PROTOCOL_LABELS)
        lines.append(
            f"Top overall & {tex_escape(feature_set)} & {tex_escape(protocol)} & "
            f"{fmt(top.hit_rate * 100.0, 1)} & {fmt(top.mean_net_bp, 3)} & {fmt(top.sr_net, 2)} & {int(top.obs)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_tree_horserace_compact(summary: pd.DataFrame, output_file: Path) -> None:
    tree_models = [
        "lgbm",
        "lgbm_deep",
        "xgb",
        "xgb_deep",
        "catboost",
        "catboost_deep",
        "lgbm_clf",
        "lgbm_clf_deep",
        "xgb_clf",
        "xgb_clf_deep",
        "catboost_clf",
        "catboost_clf_deep",
        "rf_clf",
        "et_clf",
        "hgb_clf",
    ]
    tree = summary[summary["model"].astype(str).isin(tree_models)].copy()
    if tree.empty:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("% No tree-model results available.\n", encoding="utf-8")
        return

    best = (
        tree.sort_values(["mean_net_bp", "sr_net"], ascending=[False, False])
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best = best.sort_values(["mean_net_bp", "sr_net"], ascending=[False, False]).reset_index(drop=True)
    top = best.iloc[0] if not best.empty else None

    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Model & Best Feature Set & Window & Hit Rate (\%) & Mean Net (bps) & SR Net & N \\",
        r"\midrule",
    ]
    for row in best.itertuples(index=False):
        model = display_label(row.model, MODEL_LABELS)
        feature_set = display_label(row.feature_set, FEATURE_SET_LABELS)
        protocol = display_label(row.protocol, PROTOCOL_LABELS)
        lines.append(
            f"{tex_escape(model)} & {tex_escape(feature_set)} & {tex_escape(protocol)} & "
            f"{fmt(row.hit_rate * 100.0, 1)} & {fmt(row.mean_net_bp, 3)} & {fmt(row.sr_net, 2)} & {int(row.obs)} \\\\"
        )

    lines.append(r"\midrule")
    if top is not None:
        feature_set = display_label(top.feature_set, FEATURE_SET_LABELS)
        protocol = display_label(top.protocol, PROTOCOL_LABELS)
        lines.append(
            f"Top overall & {tex_escape(feature_set)} & {tex_escape(protocol)} & "
            f"{fmt(top.hit_rate * 100.0, 1)} & {fmt(top.mean_net_bp, 3)} & {fmt(top.sr_net, 2)} & {int(top.obs)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = get_project_root(args.project_root)
    data_dir = get_data_dir(root)
    tables_dir = get_tables_dir(root)
    args.task = str(args.task)

    if args.models is None or len(args.models) == 0:
        args.models = default_model_ids(args.task)

    if not (0.0 < float(args.decision_threshold) < 1.0):
        raise ValueError(f"--decision-threshold must be in (0,1), got {args.decision_threshold}")

    if not HAS_LIGHTGBM and any(m in {"lgbm", "lgbm_deep", "lgbm_clf", "lgbm_clf_deep"} for m in args.models):
        warnings.warn(
            "lightgbm is not installed; skipping lgbm models. Install via: pip install lightgbm",
            RuntimeWarning,
        )
    if not HAS_XGBOOST and any(m in {"xgb", "xgb_deep", "xgb_clf", "xgb_clf_deep"} for m in args.models):
        warnings.warn(
            "xgboost is not installed; skipping xgb models. Install via: pip install xgboost",
            RuntimeWarning,
        )
    if not HAS_CATBOOST and any(m in {"catboost", "catboost_deep", "catboost_clf", "catboost_clf_deep"} for m in args.models):
        warnings.warn(
            "catboost is not installed; skipping catboost models. Install via: pip install catboost",
            RuntimeWarning,
        )

    data, selected_mnes = prepare_dataset(
        data_dir=data_dir,
        representative_moneyness=bool(args.representative_moneyness),
        max_moneyness_dev=float(args.max_moneyness_dev),
    )

    model_zoo = make_model_zoo(
        task=args.task,
        random_state=0,
        nn_max_epochs=args.nn_max_epochs,
        nn_patience=args.nn_patience,
    )
    scaling_specs = make_scaling_specs()
    if args.task == "binary":
        dedup_scalings: dict[str, ScalingSpec] = {}
        seen_keys: set[tuple[str, str]] = set()
        for key, spec in scaling_specs.items():
            dedup_key = (spec.feature_space, spec.ts_mode)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            dedup_scalings[key] = ScalingSpec(
                name=spec.name,
                feature_space=spec.feature_space,
                ts_mode=spec.ts_mode,
                y_mode="raw",
            )
        scaling_specs = dedup_scalings

    unknown_models = [m for m in args.models if m not in model_zoo]
    if unknown_models:
        warnings.warn(f"Unknown/unavailable models skipped: {unknown_models}", RuntimeWarning)

    unknown_scalings = [s for s in args.scalings if s not in scaling_specs]
    if unknown_scalings:
        warnings.warn(f"Unknown scaling specs skipped: {unknown_scalings}", RuntimeWarning)

    unknown_feature_sets = [s for s in args.feature_sets if s not in FEATURE_SETS]
    if unknown_feature_sets:
        warnings.warn(f"Unknown feature sets skipped: {unknown_feature_sets}", RuntimeWarning)

    models = [m for m in args.models if m in model_zoo]
    scalings = [s for s in args.scalings if s in scaling_specs]
    feature_sets = [s for s in args.feature_sets if s in FEATURE_SETS]
    protocols = [p for p in args.protocols if p in {"expanding", "rolling"}]

    rows = []
    preds_all = []

    prepared_by_key: dict[tuple[str, str], tuple[list[str], list[str], ScalingSpec, pd.DataFrame]] = {}
    for feature_set in feature_sets:
        for scaling_name in scalings:
            spec = scaling_specs[scaling_name]
            feat_cols, ts_scale_cols = select_features(data=data, spec=spec, feature_set=feature_set)
            prepared_by_key[(feature_set, scaling_name)] = (
                feat_cols,
                ts_scale_cols,
                spec,
                prepare_work(data=data, features=feat_cols),
            )

    for feature_set in feature_sets:
        for scaling_name in scalings:
            feat_cols, ts_scale_cols, spec, prepared = prepared_by_key[(feature_set, scaling_name)]
            if prepared.empty:
                continue

            for model_name in models:
                model = model_zoo[model_name]
                for protocol in protocols:
                    pred = walk_forward_predict(
                        work=prepared,
                        features=feat_cols,
                        ts_scale_cols=ts_scale_cols,
                        model=model,
                        protocol=protocol,
                        min_train_days=args.min_train_days,
                        rolling_window=args.rolling_window,
                        refit_every=args.refit_every,
                        y_mode=spec.y_mode,
                        task=args.task,
                        decision_mode=args.decision_mode,
                        decision_threshold=float(args.decision_threshold),
                    )

                    smry = summarize_predictions(pred=pred, net_cost=args.net_cost)
                    if not smry:
                        continue

                    rows.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
                            "scaling": scaling_name,
                            "protocol": protocol,
                            "representative_moneyness": bool(args.representative_moneyness),
                            "max_moneyness_dev": float(args.max_moneyness_dev),
                            "n_features": int(len(feat_cols)),
                            "feature_space": spec.feature_space,
                            "ts_mode": spec.ts_mode,
                            "y_mode": ("binary" if args.task == "binary" else spec.y_mode),
                            "task": args.task,
                            "decision_mode": args.decision_mode,
                            "decision_threshold": float(args.decision_threshold),
                            **smry,
                        }
                    )

                    if bool(args.store_preds):
                        pred = pred.copy()
                        pred["model"] = model_name
                        pred["feature_set"] = feature_set
                        pred["scaling"] = scaling_name
                        pred["protocol"] = protocol
                        pred["n_features"] = int(len(feat_cols))
                        pred["feature_space"] = spec.feature_space
                        pred["ts_mode"] = spec.ts_mode
                        pred["y_mode"] = ("binary" if args.task == "binary" else spec.y_mode)
                        pred["task"] = args.task
                        pred["decision_mode"] = args.decision_mode
                        pred["decision_threshold"] = float(args.decision_threshold)
                        preds_all.append(pred)

    if not rows:
        raise RuntimeError("No model/scaling/protocol combination produced results.")

    summary = (
        pd.DataFrame(rows)
        .sort_values(["feature_set", "model", "scaling", "protocol"])
        .reset_index(drop=True)
    )
    preds_out = pd.concat(preds_all, axis=0, ignore_index=True) if (bool(args.store_preds) and preds_all) else pd.DataFrame()

    suffix = "" if args.task == "regression" else f"_{args.task}"
    summary_out = args.summary_out or (data_dir / f"conditional_model_zoo{suffix}_summary.csv")
    pred_out = args.pred_out or (data_dir / f"conditional_model_zoo{suffix}_predictions.parquet")
    rep_moneyness_out = args.rep_moneyness_out or (data_dir / "conditional_representative_moneyness.csv")
    latex_out = args.latex_out or (tables_dir / f"0dte_conditional_model_zoo{suffix}.tex")
    latex_compact_out = (
        args.latex_compact_out
        or (tables_dir / f"0dte_conditional_model_zoo{suffix}_compact.tex")
    )
    latex_tree_compact_out = (
        args.latex_tree_compact_out
        or (tables_dir / f"0dte_conditional_model_zoo{suffix}_tree_horserace_compact.tex")
    )

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    if bool(args.store_preds):
        pred_out.parent.mkdir(parents=True, exist_ok=True)
    rep_moneyness_out.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_out, index=False)
    if bool(args.store_preds) and (not preds_out.empty):
        preds_out.to_parquet(pred_out, index=False)
    if not selected_mnes.empty:
        selected_mnes.to_csv(rep_moneyness_out, index=False)
    write_latex(summary=summary, output_file=latex_out)
    write_latex_compact(summary=summary, output_file=latex_compact_out)
    write_latex_tree_horserace_compact(summary=summary, output_file=latex_tree_compact_out)

    print(f"Task: {args.task}")
    print(
        "Decision: "
        f"{args.decision_mode} threshold={float(args.decision_threshold):.3f}"
    )
    print(f"Representative moneyness filter: {bool(args.representative_moneyness)}")
    if bool(args.representative_moneyness):
        print(f"Max |moneyness-1|: {float(args.max_moneyness_dev):.4f}")
        print(f"Representative map: {rep_moneyness_out}")
    print(f"Summary: {summary_out}")
    print(f"Predictions: {pred_out if bool(args.store_preds) else 'disabled (--no-store-preds)'}")
    print(f"LaTeX: {latex_out}")
    print(f"LaTeX compact: {latex_compact_out}")
    print(f"LaTeX tree compact: {latex_tree_compact_out}")
    print(f"Rows: {summary.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
