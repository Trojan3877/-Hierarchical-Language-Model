# src/utils/tracker.py

"""
Experiment Tracker Utility
--------------------------
Provides MLflow integration to log training metrics,
hyperparameters, and model artifacts for reproducibility.
"""

import mlflow
from utils.logger import logger


class ExperimentTracker:
    def __init__(self, experiment_name: str = "HLM-Experiments"):
        """
        Initialize MLflow experiment tracker.
        """
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"Initialized MLflow experiment: {experiment_name}")

    def log_params(self, params: dict):
        """Log hyperparameters"""
        with mlflow.start_run():
            mlflow.log_params(params)
            logger.info(f"Logged params: {params}")

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics (accuracy, loss, etc.)"""
        with mlflow.start_run(nested=True):
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"Logged metrics: {metrics}")

    def log_model(self, model, model_name: str = "hlm-model"):
        """Log PyTorch model"""
        with mlflow.start_run(nested=True):
            mlflow.pytorch.log_model(model, model_name)
            logger.info(f"Saved model to MLflow: {model_name}")
