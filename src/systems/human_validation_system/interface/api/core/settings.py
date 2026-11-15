"""
Interface Settings - Configuração da interface web da validação humana
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.config.human_validation import (
    AUTO_START_NEXT_ROUND,
    EVALUATORS,
    EVALUATORS_BY_DATASET,
    INCREMENT_SIZE,
    INTERFACE_DB_NAME,
    REVIEW_WINDOW_MINUTES,
    VALIDATION_DATASETS,
    SMTP_DEFAULT_HOST,
    SMTP_DEFAULT_PORT,
    RESEND_DEFAULT_SENDER,
)
from src.systems.human_validation_system.pipeline import HumanValidationPipeline


class InterfaceSettings:
    """
    Configurações da interface (experimentos, códigos de acesso, banco).

    Responsabilidades:
    - Definir quais experimentos (dataset -> data) a interface conduz
    - Guardar os avaliadores de cada dataset e os códigos de acesso (vindos do .env)
    - Resolver o caminho do banco SQLite (padrão: `data/validacao_humana/validacao_humana.db`)
    - Guardar destinatário e credenciais (Resend ou SMTP) do aviso por email (vindos do .env)
    """

    def __init__(
        self,
        experiments: Dict[str, str],
        results_dir: str,
        evaluator_codes: Dict[str, str],
        admin_code: str,
        db_path: Optional[str] = None,
        increment_size: int = INCREMENT_SIZE,
        auto_next_round: bool = AUTO_START_NEXT_ROUND,
        review_window_minutes: int = REVIEW_WINDOW_MINUTES,
        notify_email: Optional[str] = None,
        smtp_host: str = SMTP_DEFAULT_HOST,
        smtp_port: int = SMTP_DEFAULT_PORT,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        resend_api_key: Optional[str] = None,
        resend_sender: str = RESEND_DEFAULT_SENDER,
        evaluators_by_dataset: Optional[Dict[str, List[str]]] = None,
    ):
        if not admin_code:
            raise ValueError("Código do administrador ausente (HV_ADMIN_CODE)")
        mapping = evaluators_by_dataset or EVALUATORS_BY_DATASET
        unmapped = [d for d in experiments if d not in mapping]
        if unmapped:
            raise ValueError(f"Datasets sem avaliadores definidos em EVALUATORS_BY_DATASET: {unmapped}")
        # Só os avaliadores dos experimentos conduzidos precisam de código
        self.evaluators_by_dataset = {d: list(mapping[d]) for d in experiments}
        self.evaluators = list(dict.fromkeys(e for names in self.evaluators_by_dataset.values() for e in names))
        missing = [self.code_variable(e) for e in self.evaluators if not evaluator_codes.get(e)]
        if missing:
            raise ValueError(f"Códigos de acesso ausentes no .env: {missing}")

        self.experiments = experiments
        self.results_dir = str(results_dir)
        self.evaluator_codes = {e: evaluator_codes[e] for e in self.evaluators}
        self.admin_code = admin_code
        self.db_path = Path(db_path) if db_path else HumanValidationPipeline.validation_root(results_dir) / INTERFACE_DB_NAME
        self.increment_size = increment_size
        self.auto_next_round = auto_next_round
        self.review_window_minutes = review_window_minutes
        self.notify_email = notify_email
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.resend_api_key = resend_api_key
        self.resend_sender = resend_sender
        logger.debug(f"InterfaceSettings: {list(experiments)} | banco em {self.db_path}")

    def evaluators_for(self, dataset: str) -> List[str]:
        """Os 3 avaliadores do dataset."""
        return self.evaluators_by_dataset[dataset]

    @staticmethod
    def code_variable(evaluator: str) -> str:
        """Variável do .env com o código do avaliador (ex.: HV_EVALUATOR_BOOKS_1_CODE)."""
        return f"HV_EVALUATOR_{evaluator.upper()}_CODE"

    @staticmethod
    def experiment_variable(dataset: str) -> str:
        """Variável do .env com a data do experimento (ex.: HV_EXPERIMENT_BOOKS)."""
        return f"HV_EXPERIMENT_{dataset.upper()}"

    @classmethod
    def experiments_from_env(cls) -> Dict[str, str]:
        """Datasets de VALIDATION_DATASETS que têm data definida no ambiente."""
        dates = {d: os.getenv(cls.experiment_variable(d), "").strip() for d in VALIDATION_DATASETS}
        return {d: date for d, date in dates.items() if date}

    @classmethod
    def from_env(
        cls,
        experiments: Optional[Dict[str, str]] = None,
        results_dir: Optional[str] = None,
        increment_size: Optional[int] = None,
    ) -> "InterfaceSettings":
        """Lê HV_* do ambiente; argumentos explícitos têm prioridade."""
        return cls(
            experiments=experiments or cls.experiments_from_env(),
            results_dir=results_dir or os.getenv("HV_RESULTS_DIR", "data/results"),
            evaluator_codes={e: os.getenv(cls.code_variable(e), "") for e in EVALUATORS},
            admin_code=os.getenv("HV_ADMIN_CODE", ""),
            db_path=os.getenv("HV_DB_PATH") or None,
            increment_size=increment_size or int(os.getenv("HV_INCREMENT_SIZE", INCREMENT_SIZE)),
            notify_email=os.getenv("HV_NOTIFY_EMAIL") or None,
            smtp_host=os.getenv("HV_SMTP_HOST", SMTP_DEFAULT_HOST),
            smtp_port=int(os.getenv("HV_SMTP_PORT", SMTP_DEFAULT_PORT)),
            smtp_user=os.getenv("HV_SMTP_USER") or None,
            # Senhas de app do Gmail são exibidas com espaços; o SMTP aceita sem eles
            smtp_password=(os.getenv("HV_SMTP_PASSWORD") or "").replace(" ", "") or None,
            resend_api_key=os.getenv("HV_RESEND_API_KEY") or None,
            resend_sender=os.getenv("HV_RESEND_FROM") or RESEND_DEFAULT_SENDER,
        )
