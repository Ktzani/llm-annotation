from pathlib import Path
from typing import Optional

import transformers.trainer
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


class CheckpointResume(TrainerCallback):
    """
    Retomada do treino a partir dos checkpoints já salvos na pasta do fold
    Responsabilidades: achar o último checkpoint, liberar o torch.load deles e reapontar o melhor checkpoint para a pasta atual
    """

    @staticmethod
    def last_checkpoint(output_dir: str | Path) -> Optional[str]:
        """Último checkpoint-* da pasta (None se o fold ainda não salvou nenhum)"""
        return get_last_checkpoint(str(output_dir)) if Path(output_dir).is_dir() else None

    @staticmethod
    def allow_local_torch_load() -> None:
        """transformers>=4.50 barra torch.load com torch<2.6 (CVE-2025-32434, arquivos de terceiros); os checkpoints são do próprio treino"""
        transformers.trainer.check_torch_load_is_safe = lambda: None

    def on_train_begin(self, args, state, control, **kwargs):
        # O trainer_state guarda o caminho absoluto da execução original (ex.: /app/data/... no Docker)
        if state.best_model_checkpoint:
            state.best_model_checkpoint = str(Path(args.output_dir) / Path(state.best_model_checkpoint).name)
