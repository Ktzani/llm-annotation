from typing import Dict
from src.api.schemas.experiment import ExperimentStatus

# 📌 Depois isso vira Redis ou DynamoDB sem mudar API.
experiments: Dict[str, ExperimentStatus] = {}
