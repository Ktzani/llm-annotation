"""
Pipeline de preparação da VALIDAÇÃO HUMANA em rodadas sucessivas.

Segue o procedimento iterativo guiado por confiança (Merlo et al., CIKM'25):
uma rodada inicial de 30 documentos por grupo de concordância (A/B/C) e
incrementos sem reposição até a margem de erro da estimativa ficar aceitável.
A amostragem é estratificada por classe de referência dentro de cada grupo.

Como pedir uma rodada:
    HumanValidationPipeline(config).run()                 próxima rodada
    HumanValidationPipeline(config).run(round_number=k)   reproduz a rodada k já entregue
O tamanho do próximo incremento vem de `config.increment_size` e fica gravado
no registro, então pode mudar entre rodadas (ex.: 10 -> 20).

A partir da 2ª rodada, só os grupos que ainda não atingiram o critério de
parada são sorteados: o status vem da estimativa da última rodada
(`run_human_validation_estimate.py`), que precisa existir antes do pedido.

Estrutura de saída (em ``<results>/<dataset>/<date>/validacao_humana/``):
    amostragem.json                        Registro: semente, sha do CSV de origem, rodadas entregues
    guia_avaliador.md                      Guia do avaliador
    exemplos_guia.csv                      Documentos reservados para o guia (fora de todas as rodadas)
    rodada_XX/planilha_avaliacao_<avaliador>.xlsx
    rodada_XX/gabarito.csv
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.config.datasets_collected import LABEL_MEANINGS
from src.config.human_validation import (
    CLASS_DEFINITIONS,
    EVALUATORS,
    EXAMPLE_LENGTH_QUANTILE,
    EXAMPLE_MAX_CHARS,
    EXAMPLES_PER_CLASS,
    HUMAN_VALIDATION_SEED,
    ID_PREFIXES,
    INCREMENT_SIZE,
    INITIAL_ROUND_SIZE,
    INSUFFICIENT_INFO_OPTION,
    MIN_PER_CLASS_FIRST_ROUND,
)
from src.systems.human_validation_system.estimation.stopping_status import StoppingStatusReader
from src.systems.human_validation_system.output.anonymizer import AnonymousIdGenerator
from src.systems.human_validation_system.output.answer_key_writer import AnswerKeyWriter
from src.systems.human_validation_system.output.evaluation_sheet_writer import EvaluationSheetWriter
from src.systems.human_validation_system.output.evaluator_guide_writer import EvaluatorGuideWriter
from src.systems.human_validation_system.sampling.agreement_grouper import AgreementGrouper
from src.systems.human_validation_system.sampling.guide_example_selector import GuideExampleSelector
from src.systems.human_validation_system.sampling.incremental_sampler import IncrementalStratifiedSampler
from src.systems.human_validation_system.sampling.proportional_allocator import ProportionalAllocator
from src.systems.human_validation_system.sampling.sampling_ledger import SamplingLedger
from src.systems.human_validation_system.sampling.seeded_hasher import SeededHasher

DEFAULT_RESULTS_DIR = "C:\\Users\\gabri\\Documents\\GitHub\\llm-annotation\\data\\results"


class HumanValidationConfig:
    """Configurações da preparação da validação humana."""

    def __init__(
        self,
        dataset_name: str,
        specific_date: str,
        results_dir: str = DEFAULT_RESULTS_DIR,
        seed: int = HUMAN_VALIDATION_SEED,
        initial_size: int = INITIAL_ROUND_SIZE,
        increment_size: int = INCREMENT_SIZE,
        min_per_class: int = MIN_PER_CLASS_FIRST_ROUND,
        evaluators: Optional[List[str]] = None,
        examples_per_class: int = EXAMPLES_PER_CLASS,
        overwrite_guide: bool = False,
        skip_stopped_groups: bool = True,
    ):
        self.dataset_name = dataset_name
        # Data explícita: `latest` (por mtime) pode apontar para outro experimento
        self.specific_date = specific_date
        self.results_dir = results_dir
        self.seed = seed
        self.initial_size = initial_size
        self.increment_size = increment_size
        self.min_per_class = min_per_class
        self.evaluators = evaluators or EVALUATORS
        self.examples_per_class = examples_per_class
        self.overwrite_guide = overwrite_guide
        # Não sorteia grupos com status `parar` na última estimativa
        self.skip_stopped_groups = skip_stopped_groups


class HumanValidationPipeline:
    """Pipeline principal da preparação da validação humana."""

    OUTPUT_DIR_NAME = "validacao_humana"

    def __init__(self, config: HumanValidationConfig):
        self.config = config
        self.results_dataset_path = Path(config.results_dir) / config.dataset_name / config.specific_date
        self.output_dir = self.results_dataset_path / self.OUTPUT_DIR_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.label_names = {int(k): v for k, v in LABEL_MEANINGS[config.dataset_name].items()}

        # Inicializar componentes
        hasher = SeededHasher(config.seed, config.dataset_name)
        self.grouper = AgreementGrouper()
        self.sampler = IncrementalStratifiedSampler(hasher, ProportionalAllocator())
        self.example_selector = GuideExampleSelector(hasher, config.examples_per_class, EXAMPLE_LENGTH_QUANTILE)
        self.anonymizer = AnonymousIdGenerator(
            hasher, ID_PREFIXES.get(config.dataset_name, config.dataset_name[:2].upper())
        )
        self.sheet_writer = EvaluationSheetWriter(
            [self.label_names[c] for c in sorted(self.label_names)], INSUFFICIENT_INFO_OPTION
        )
        self.guide_writer = EvaluatorGuideWriter(
            config.dataset_name,
            self.label_names,
            CLASS_DEFINITIONS.get(config.dataset_name, {}),
            EXAMPLE_MAX_CHARS,
            INSUFFICIENT_INFO_OPTION,
        )
        self.ledger = SamplingLedger(self.output_dir)
        self.stopping_status = StoppingStatusReader()
        logger.success(f"✓ Setup completo — saída em: {self.output_dir}")

    @property
    def source_path(self) -> Path:
        return self.results_dataset_path / "consensus" / "dataset_consenso.csv"

    def load_consensus_data(self) -> pd.DataFrame:
        if not self.source_path.exists():
            raise FileNotFoundError(f"Dataset de consenso não encontrado: {self.source_path}")
        df = pd.read_csv(self.source_path)
        logger.info(f"Carregado: {len(df)} instâncias de {self.source_path}")
        return df

    def _prepare_ledger(self, df: pd.DataFrame, examples: pd.DataFrame, sampling_pool: pd.DataFrame) -> None:
        """Cria o registro ou confere que semente, origem e exemplos não mudaram."""
        source_sha = self.ledger.file_sha256(self.source_path)
        example_ids = examples["text_id"].tolist()

        if not self.ledger.exists():
            llm_columns = self.grouper.detect_llm_columns(df)
            self.ledger.create(
                self.config.dataset_name,
                self.config.seed,
                self.source_path,
                source_sha,
                {f"anotacao_llm_{i}": c for i, c in enumerate(llm_columns, start=1)},
                example_ids,
                self.config.min_per_class,
                self.sampler.strata_sizes(sampling_pool),
            )
            return

        self.ledger.load()
        self.ledger.check_compatible(self.config.seed, source_sha)
        if example_ids != self.ledger.guide_example_ids:
            raise ValueError("Exemplos do guia recalculados diferem dos registrados")

    def _verify_replay(self, pool: pd.DataFrame, up_to: int) -> None:
        """Confere que a semente reproduz exatamente as rodadas registradas."""
        if up_to == 0:
            return
        replayed = self.sampler.replay(
            pool, self.ledger.sizes(up_to), self.ledger.min_per_class, self.ledger.round_groups(up_to)
        )
        recorded = [[row["text_id"] for row in r["rows"]] for r in self.ledger.rounds[:up_to]]
        if replayed != recorded:
            raise ValueError("A sequência recalculada difere das rodadas registradas")
        logger.info(f"Rodadas 1..{up_to} reproduzidas a partir da semente {self.config.seed}")

    def _active_groups(self, round_number: int) -> List[str]:
        """Grupos a sortear: todos na 1ª rodada; depois, os que ainda não pararam."""
        groups = list(self.sampler.GROUPS)
        if round_number == 1 or not self.config.skip_stopped_groups:
            return groups
        stopped = self.stopping_status.stopped_groups(self.output_dir, round_number - 1)
        if stopped:
            logger.info(f"Grupos que já atingiram o critério de parada (não sorteados): {stopped}")
        return [g for g in groups if g not in stopped]

    def _draw_round(self, pool: pd.DataFrame, round_number: int, groups: List[str], anon_ids: Dict[str, str]) -> None:
        """Sorteia a próxima rodada e a acrescenta ao registro."""
        size = self.config.initial_size if round_number == 1 else self.config.increment_size
        selected, allocation = self.sampler.next_round(
            pool, self.ledger.delivered_ids(), size, self.ledger.min_per_class if round_number == 1 else 0, groups
        )
        ordered = self.sampler.shuffle(selected, round_number)
        rows = [
            {"text_id": t, "id_anonimo": anon_ids[t], "grupo": g}
            for t, g in zip(ordered["text_id"], ordered["grupo"])
        ]
        self.ledger.append_round(size, groups, allocation, rows)

        logger.info(f"Rodada {round_number} sorteada ({size} por grupo ativo: {groups}):")
        for group, alloc in allocation.items():
            logger.info(f"  Grupo {group}: {sum(alloc.values())} -> {alloc}")

    def _write_round(self, round_number: int, pool: pd.DataFrame) -> Path:
        """Gera planilhas e gabarito a partir do registro da rodada."""
        record = self.ledger.get_round(round_number)
        round_dir = self.output_dir / f"rodada_{round_number:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        ids = [row["text_id"] for row in record["rows"]]
        sample = pool.set_index("text_id").loc[ids].reset_index()
        sample["id_anonimo"] = [row["id_anonimo"] for row in record["rows"]]

        sheet_rows = sample[["id_anonimo", "text"]].rename(columns={"text": "texto"})
        self.sheet_writer.write_copies(sheet_rows, self.config.evaluators, round_dir)

        llm_columns = list(self.ledger.data["llm_columns"].values())
        AnswerKeyWriter(self.label_names, llm_columns).write(sample, round_dir)
        return round_dir

    def run(self, round_number: Optional[int] = None) -> Optional[Path]:
        """Gera a próxima rodada (None) ou reproduz a rodada `round_number`; None se todos os grupos pararam."""
        logger.info("=" * 60)
        logger.info(f"Validação humana — {self.config.dataset_name}")
        logger.info("=" * 60)

        df = self.load_consensus_data()
        pool = self.grouper.assign(df)

        # Exemplos saem do pool antes de qualquer sorteio: ficam fora de todas as rodadas
        examples = self.example_selector.select(pool)
        sampling_pool = pool[~pool["text_id"].isin(examples["text_id"])].reset_index(drop=True)
        self._prepare_ledger(df, examples, sampling_pool)

        examples.to_csv(self.output_dir / "exemplos_guia.csv", index=False)
        self.guide_writer.write(examples, self.output_dir, overwrite=self.config.overwrite_guide)

        last = self.ledger.last_round
        target = round_number or last + 1
        if not 1 <= target <= last + 1:
            raise ValueError(f"Rodada {target} inválida: já entregues {last}; a próxima é {last + 1}")

        self._verify_replay(sampling_pool, min(target, last))
        if target == last + 1:
            groups = self._active_groups(target)
            if not groups:
                logger.success("Todos os grupos atingiram o critério de parada: nenhuma rodada nova")
                return None
            self._draw_round(sampling_pool, target, groups, self.anonymizer.generate(pool["text_id"]))
        else:
            logger.info(f"Rodada {target} já entregue: regenerando a partir do registro")

        round_dir = self._write_round(target, sampling_pool)
        delivered = len(self.ledger.delivered_ids())
        logger.success(f"Rodada {target} pronta em {round_dir} ({delivered} documentos entregues no total)")
        return round_dir
