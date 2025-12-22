import mlflow
import mlflow.pytorch

mlflow.set_experiment("Hierarchy-LLM-Experiment")

def log_model(model, metrics: dict):
    with mlflow.start_run():
        for k,v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.pytorch.log_model(model, "llm_model")
