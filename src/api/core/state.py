from typing import Dict
from src.api.schemas.annotation_experiment.experiment import AnnotationStatus
from src.api.schemas.consensus.consensus import ConsensusStatus
from src.api.schemas.fine_tuning.fine_tuning import FineTuningStatus
from src.api.core.job_runner import JobRunner

# 📌 Depois isso vira Redis ou DynamoDB sem mudar API.
annotation_jobs: Dict[str, AnnotationStatus] = {}

fine_tuning_jobs: Dict[str, FineTuningStatus] = {}

consensus_jobs: Dict[str, ConsensusStatus] = {}

# Jobs em execução (anotação e fine-tuning), para o cancelamento
job_runner = JobRunner()
