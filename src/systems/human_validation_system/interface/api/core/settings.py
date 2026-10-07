"""
Interface Settings - Configuração da interface web da validação humana
"""
import os
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from src.config.human_validation import (
    AUTO_START_NEXT_ROUND,
    EVALUATORS,
    INCREMENT_SIZE,
    INTERFACE_DB_NAME,
    REVIEW_WINDOW_MINUTES,
    SMTP_DEFAULT_HOST,
    SMTP_DEFAULT_PORT,
)
from src.systems.human_validation_system.pipeline import HumanValidationPipeline


class InterfaceSettings:
    """
    Configurações da interface (experimentos, códigos de acesso, banco).

    Responsabilidades:
    - Definir quais experimentos (dataset -> data) a interface conduz
    - Guardar os códigos de acesso de avaliadores e administrador (vindos do .env)
    - Resolver o caminho do banco SQLite (padrão: `data/validacao_humana/validacao_humana.db`)
    - Guardar destinatário e credenciais SMTP do aviso por email (vindos do .env)
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
    ):
        if not admin_code:
            raise ValueError("Código do administrador ausente (HV_ADMIN_CODE)")
        missing = [self.code_variable(e) for e in EVALUATORS if not evaluator_codes.get(e)]
        if missing:
            raise ValueError(f"Códigos de acesso ausentes no .env: {missing}")

        self.experiments = experiments
        self.results_dir = str(results_dir)
        self.evaluators = list(EVALUATORS)
        self.evaluator_codes = {e: evaluator_codes[e] for e in EVALUATORS}
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
        logger.debug(f"InterfaceSettings: {list(experiments)} | banco em {self.db_path}")

    @staticmethod
    def code_variable(evaluator: str) -> str:
        """Variável do .env com o código do avaliador (ex.: HV_CODE_AVALIADOR_1)."""
        return f"HV_CODE_{evaluator.upper()}"

    @staticmethod
    def parse_pairs(raw: str) -> Dict[str, str]:
        """'a:1,b:2' -> {'a': '1', 'b': '2'}."""
        pairs = (item.split(":", 1) for item in raw.split(",") if item.strip())
        return {k.strip(): v.strip() for k, v in pairs}

    @classmethod
    def from_env(
        cls,
        experiments: Optional[Dict[str, str]] = None,
        results_dir: Optional[str] = None,
        increment_size: Optional[int] = None,
    ) -> "InterfaceSettings":
        """Lê HV_* do ambiente; argumentos explícitos têm prioridade."""
        return cls(
            experiments=experiments or cls.parse_pairs(os.getenv("HV_EXPERIMENTS", "")),
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
        )
