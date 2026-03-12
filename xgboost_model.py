"""
XGBoost Baseline
Reference: Chen & Guestrin "XGBoost: A Scalable Tree Boosting System" (2016)
Following the stock forecasting framework:
  1) Flatten sequential input (N, T, F) -> (N, T*F) for tabular use
  2) XGBoost regressor fits on flattened features
  3) Outputs prediction (N,)
Input: (N, T, F) -> Output: (N,)
"""

import numpy as np
import pandas as pd
import pickle
from typing import Optional

import torch
from base_model import SequenceModel, calc_ic, drop_extreme, zscore
from torch.utils.data import DataLoader

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class FeatureFlattener:
    """
    Flattens sequential stock features for tabular models.
    (N, T, F) -> (N, T*F) or (N, F) when use_last_step only.
    """
    def __init__(self, use_last_step: bool = False):
        self.use_last_step = use_last_step

    def transform(self, x: np.ndarray) -> np.ndarray:
        # x: (N, T, F)
        if self.use_last_step:
            return x[:, -1, :]   # (N, F)
        return x.reshape(x.shape[0], -1)   # (N, T*F)


class XGBoostNet:
    """
    XGBoost regressor with optional feature flattening.
    Wraps sklearn-style API for use in the sequence-model pipeline.
    """
    def __init__(
        self,
        d_feat: int,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
        use_last_step: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.d_feat = d_feat
        self.use_last_step = use_last_step
        self.flattener = FeatureFlattener(use_last_step=use_last_step)
        self._model = None
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": seed,
            **kwargs,
        }

    def _ensure_model(self):
        if xgb is None:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        if self._model is None:
            self._model = xgb.XGBRegressor(**self._params)
        return self._model

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs):
        X_flat = self.flattener.transform(X)
        model = self._ensure_model()
        model.fit(X_flat, y, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self.flattener.transform(X)
        return self._ensure_model().predict(X_flat)

    def get_booster(self):
        return self._ensure_model().get_booster()


class XGBoostModel(SequenceModel):
    """
    XGBoost baseline that conforms to the SequenceModel API.
    Flattens (N, T, F) batches and trains a single XGB regressor.
    """
    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_last_step: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_last_step = use_last_step
        self.init_model()

    def init_model(self):
        self.model = XGBoostNet(
            d_feat=self.d_feat,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            use_last_step=self.use_last_step,
            seed=self.seed,
        )
        # Skip super().init_model() — no PyTorch optimizer / device
        self.fitted = -1
        print(f"[Check] XGBoost model initialized (d_feat={self.d_feat}, use_last_step={self.use_last_step})")

    def loss_fn(self, pred, label):
        mask = ~np.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2
        return float(np.mean(loss))

    def _collect_train_batches(self, data_loader):
        """Collect (X, y) from data loader with drop_extreme and zscore on labels."""
        X_list, y_list = [], []
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].numpy()
            label = data[:, -1, -1].numpy()
            mask, label_t = drop_extreme(torch.from_numpy(label))
            mask = mask.numpy()
            feature = feature[mask, :, :]
            label_t = zscore(label_t).numpy()
            X_list.append(feature)
            y_list.append(label_t)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        X_train, y_train = self._collect_train_batches(train_loader)

        self.model.fit(X_train, y_train)

        if dl_valid is not None:
            valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=False)
            preds, labels = [], []
            for data in valid_loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, 0:-1].numpy()
                label = data[:, -1, -1].numpy()
                preds.append(self.model.predict(feature).ravel())
                labels.append(label)
            pred_valid = np.concatenate(preds)
            label_valid = np.concatenate(labels)
            daily_ic, daily_ric = calc_ic(
                torch.from_numpy(pred_valid),
                torch.from_numpy(label_valid),
            )
            ic, ric = float(daily_ic), float(daily_ric)
            icir = ic / (np.std([ic]) + 1e-8)
            ricir = ric / (np.std([ric]) + 1e-8)
            print(f"Epoch 0, valid IC {ic:.4f}, ICIR {icir:.3f}, RIC {ric:.4f}, RICIR {ricir:.3f} (single XGBoost fit).")
        else:
            print("XGBoost fit completed (no validation).")

        self.fitted = 0
        param_path = f"{self.save_path}/{self.save_prefix}_{self.seed}.pkl"
        with open(param_path, "wb") as f:
            pickle.dump({"model": self.model._model, "flattener": self.model.flattener}, f)
        print(f"Model saved to {param_path}")

    def load_param(self, param_path: str):
        with open(param_path, "rb") as f:
            state = pickle.load(f)
        self.model._model = state["model"]
        self.model.flattener = state.get("flattener", self.model.flattener)
        self.fitted = 999

    def predict(self, dl_test):
        if self.fitted < 0:
            raise ValueError("model is not fitted yet!")
        print("Epoch:", self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        preds = []
        ic_vals = []
        ric_vals = []
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].numpy()
            label = data[:, -1, -1].numpy()
            pred = self.model.predict(feature).ravel()
            preds.append(pred)
            daily_ic, daily_ric = calc_ic(torch.from_numpy(pred), torch.from_numpy(label))
            ic_vals.append(float(daily_ic))
            ric_vals.append(float(daily_ric))

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())
        metrics = {
            "IC": np.mean(ic_vals),
            "ICIR": np.mean(ic_vals) / (np.std(ic_vals) + 1e-8),
            "RIC": np.mean(ric_vals),
            "RICIR": np.mean(ric_vals) / (np.std(ric_vals) + 1e-8),
        }
        return predictions, metrics
