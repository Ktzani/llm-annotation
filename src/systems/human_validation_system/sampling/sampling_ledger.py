"""
Sampling Ledger - Registro append-only das rodadas entregues (amostragem.json)
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger


class SamplingLedger:
    """
    Registra semente, origem dos dados e rodadas já entregues.

    O sorteio é determinístico, mas o registro é o que congela o que foi
    entregue: se o CSV de origem ou a semente mudarem, a sequência recalculada
    seria outra, então o pipeline recusa continuar em vez de divergir.

    Responsabilidades:
    - Criar, carregar e salvar o `amostragem.json`
    - Conferir semente e sha256 do CSV de origem
    - Acrescentar rodadas (nunca reescreve uma rodada existente)
    """

    FILE_NAME = "amostragem.json"
    ALL_GROUPS = ("A", "B", "C")

    def __init__(self, output_dir: Path):
        self.path = Path(output_dir) / self.FILE_NAME
        self.data: Dict = {}
        logger.debug(f"SamplingLedger inicializado: {self.path}")

    @staticmethod
    def file_sha256(path: Path) -> str:
        """sha256 do arquivo de origem."""
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logger.info(f"Registro carregado: {len(self.rounds)} rodada(s) entregue(s)")

    def create(
        self,
        dataset_name: str,
        seed: int,
        source_path: Path,
        source_sha256: str,
        llm_columns: Dict[str, str],
        guide_example_ids: List[str],
        min_per_class: int,
        strata_sizes: Dict[str, Dict[int, int]],
    ) -> None:
        self.data = {
            "dataset": dataset_name,
            "seed": seed,
            "source": {"path": str(source_path), "sha256": source_sha256},
            "llm_columns": llm_columns,
            "guide_example_ids": guide_example_ids,
            "min_per_class_first_round": min_per_class,
            # N_h da população amostrada (sem os exemplos do guia): pesos W_h do estimador
            "strata_sizes": {g: {str(c): n for c, n in s.items()} for g, s in strata_sizes.items()},
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rounds": [],
        }
        self.save()

    def check_compatible(self, seed: int, source_sha256: str) -> None:
        """Erro se semente ou dados de origem diferem do registrado."""
        if self.data["seed"] != seed:
            raise ValueError(f"Semente {seed} difere da registrada ({self.data['seed']}) em {self.path}")
        if self.data["source"]["sha256"] != source_sha256:
            raise ValueError(
                f"O CSV de origem mudou desde o registro ({self.data['source']['path']}). "
                f"As rodadas entregues não seriam reproduzíveis; use outra pasta de saída."
            )

    @property
    def rounds(self) -> List[Dict]:
        return self.data.get("rounds", [])

    @property
    def last_round(self) -> int:
        return len(self.rounds)

    @property
    def guide_example_ids(self) -> List[str]:
        return self.data["guide_example_ids"]

    @property
    def min_per_class(self) -> int:
        return self.data["min_per_class_first_round"]

    def get_round(self, round_number: int) -> Optional[Dict]:
        return self.rounds[round_number - 1] if 1 <= round_number <= self.last_round else None

    def sizes(self, up_to: int) -> List[int]:
        return [r["size_per_group"] for r in self.rounds[:up_to]]

    def round_groups(self, up_to: int) -> List[List[str]]:
        """Grupos sorteados em cada rodada (grupos que já pararam ficam de fora)."""
        return [r.get("groups", list(self.ALL_GROUPS)) for r in self.rounds[:up_to]]

    def delivered_ids(self) -> Set[str]:
        return {row["text_id"] for r in self.rounds for row in r["rows"]}

    def append_round(self, size: int, groups: List[str], allocation: Dict, rows: List[Dict]) -> Dict:
        record = {
            "round": self.last_round + 1,
            "size_per_group": size,
            "groups": list(groups),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "allocation": {g: {str(c): n for c, n in a.items()} for g, a in allocation.items()},
            "rows": rows,
        }
        self.data["rounds"].append(record)
        self.save()
        return record

    def save(self) -> None:
        """Escrita atômica (arquivo temporário + replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)
