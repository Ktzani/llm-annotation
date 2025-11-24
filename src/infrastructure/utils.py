import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.infrastructure.config_credentials_env import credentials_handler
import logging

def configure_logging():
    """
    Configura o logging global da aplicação.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

configure_logging()

queue_training = getattr(credentials_handler, "QUEUE_TRAINING_GT589", None)
queue_predict = getattr(credentials_handler, "QUEUE_PREDICT", None)
s3_mlflow = getattr(credentials_handler, "MLFLOW_S3", None)
s3_mlflow_old = getattr(credentials_handler, "MLFLOW_S3_OLD", None)
