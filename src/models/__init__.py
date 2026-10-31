"""Model training and prediction interfaces for XGBoost and MLP."""

from src.models.train import DEFAULT_CONFIGS, MODEL_NAMES, TrainedModel, split_xy, train_model

__all__ = ["DEFAULT_CONFIGS", "MODEL_NAMES", "TrainedModel", "split_xy", "train_model"]
