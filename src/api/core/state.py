from typing import Dict
from src.api.schemas.annotation_experiment.experiment import ExperimentStatus
from src.api.schemas.consensus.consensus import ConsensusStatus
from src.api.core.job_runner import JobRunner

# 📌 Depois isso vira Redis ou DynamoDB sem mudar API.
experiments: Dict[str, ExperimentStatus] = {}

fine_tuning_jobs: Dict[str, ExperimentStatus] = {}

consensus_jobs: Dict[str, ConsensusStatus] = {}

# Jobs em execução (experimentos e fine-tuning), para o cancelamento
job_runner = JobRunner()
