from typing import Dict
from src.api.schemas.annotation_experiment.experiment import ExperimentStatus
from src.api.core.cancellation import CancellationToken

# 📌 Depois isso vira Redis ou DynamoDB sem mudar API.
experiments: Dict[str, ExperimentStatus] = {}

fine_tuning_jobs: Dict[str, ExperimentStatus] = {}

# Um token por job/experimento em execução
cancellation_tokens: dict[str, CancellationToken] = {}