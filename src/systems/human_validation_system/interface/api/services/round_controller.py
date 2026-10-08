"""
Round Controller - Abre, acompanha e fecha rodadas da validação humana via interface
"""
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.config.datasets_collected import LABEL_MEANINGS
from src.config.human_validation import (
    CLASS_DEFINITIONS,
    EXAMPLE_MAX_CHARS,
    INSUFFICIENT_INFO_OPTION,
    PRIMARY_METRIC,
)
from src.systems.human_validation_system.estimation.pipeline import (
    HumanValidationEstimationConfig,
    HumanValidationEstimationPipeline,
)
from src.systems.human_validation_system.estimation.situation_classifier import SituationClassifier
from src.systems.human_validation_system.estimation.stopping_status import StoppingStatusReader
from src.systems.human_validation_system.interface.api.services.consensus_file_manager import ConsensusFileManager
from src.systems.human_validation_system.interface.api.services.consolidated_workbook import ConsolidatedWorkbookWriter
from src.systems.human_validation_system.interface.api.services.database_response_source import DatabaseResponseSource
from src.systems.human_validation_system.interface.api.services.email_notifier import EmailNotifier
from src.systems.human_validation_system.interface.api.services.response_store import ResponseStore
from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.pipeline import HumanValidationConfig, HumanValidationPipeline
from src.systems.human_validation_system.sampling.sampling_ledger import SamplingLedger


class RoundStateError(RuntimeError):
    """Operação incompatível com o estado da rodada (ex.: abrir com rodada aberta)."""


class RoundIncompleteError(RoundStateError):
    """Tentativa de fechar a rodada com avaliador incompleto."""


class RoundController:
    """
    Conduz o ciclo de rodadas de cada dataset.

    Responsabilidades:
    - Abrir a próxima rodada chamando a amostragem e publicando só id_anonimo + texto
    - Informar o progresso de cada avaliador
    - Fechar a rodada somente com todos completos: estimação (respostas do banco),
      critério de parada e planilha consolidada
    - Abrir a próxima rodada automaticamente se algum grupo não parou
    - Fechar sozinho as rodadas completas após a janela de revisão (close_due_rounds)
    - As duas automações podem ser ligadas/desligadas pelo administrador (salvas no banco)
    - Guardar as métricas acumuladas até cada rodada para o painel por dataset
    - Avisar o administrador por email quando uma rodada fecha
    - Receber o CSV de consenso pela interface (bloqueado depois do 1º sorteio)
    - Reiniciar um dataset ou tudo, guardando antes uma cópia de segurança
    """

    def __init__(self, settings: InterfaceSettings, store: ResponseStore, notifier: Optional[EmailNotifier] = None):
        self.settings = settings
        self.store = store
        self.workbook_writer = ConsolidatedWorkbookWriter()
        self.consensus = ConsensusFileManager(settings.results_dir)
        self.notifier = notifier or EmailNotifier(
            settings.notify_email, settings.smtp_host, settings.smtp_port, settings.smtp_user, settings.smtp_password
        )
        # Serializa abrir/fechar (botão do admin x verificação automática); reentrante pois fechar abre a próxima
        self._lock = threading.RLock()
        logger.debug(f"RoundController inicializado ({list(settings.experiments)})")

    # ------------------------------------------------------------- caminhos
    def _check_dataset(self, dataset: str) -> None:
        if dataset not in self.settings.experiments:
            raise KeyError(f"Dataset não configurado: {dataset}")

    def store_key(self, dataset: str) -> str:
        """Chave do experimento no banco (dataset + data): rodadas de testes ou de outra data nunca se misturam."""
        return f"{dataset}/{self.settings.experiments[dataset]}"

    def validation_dir(self, dataset: str) -> Path:
        return HumanValidationPipeline.validation_dir(self.settings.results_dir, dataset, self.settings.experiments[dataset])

    def workbook_path(self, dataset: str) -> Path:
        return self.validation_dir(dataset) / f"validacao_consolidada_{dataset}.xlsx"

    def class_names(self, dataset: str) -> List[str]:
        meanings = LABEL_MEANINGS[dataset]
        return [meanings[k] for k in sorted(meanings, key=int)]

    def _sampling_config(self, dataset: str) -> HumanValidationConfig:
        return HumanValidationConfig(
            dataset_name=dataset,
            specific_date=self.settings.experiments[dataset],
            results_dir=self.settings.results_dir,
            increment_size=self.settings.increment_size,
            evaluators=self.settings.evaluators,
            export_sheets=False,
        )

    # ------------------------------------------------------------- guia
    def guide(self, dataset: str) -> Dict:
        """Classes (nome + descrição de apoio) e exemplos reservados do guia."""
        self._check_dataset(dataset)
        meanings = LABEL_MEANINGS[dataset]
        definitions = CLASS_DEFINITIONS.get(dataset, {})
        examples: Dict[str, List[str]] = {name: [] for name in meanings.values()}

        path = self.validation_dir(dataset) / "exemplos_guia.csv"
        if path.exists():
            for _, row in pd.read_csv(path).iterrows():
                text = " ".join(str(row["text"]).split())
                if len(text) > EXAMPLE_MAX_CHARS:
                    text = text[:EXAMPLE_MAX_CHARS].rstrip() + "…"
                examples[meanings[str(int(row["ground_truth"]))]].append(text)

        return {
            "classes": [
                {"rotulo": meanings[k], "descricao": definitions.get(k, ""), "exemplos": examples[meanings[k]]}
                for k in sorted(meanings, key=int)
            ],
            "opcao_indecidivel": INSUFFICIENT_INFO_OPTION,
        }

    # ------------------------------------------------------------- status
    def progress(self, dataset: str, round_number: int) -> Dict[str, Dict[str, int]]:
        total = self.store.round_size(self.store_key(dataset), round_number)
        answered = self.store.progress(self.store_key(dataset), round_number, self.settings.evaluators)
        return {e: {"respondidos": n, "total": total, "faltam": total - n} for e, n in answered.items()}

    def last_result(self, dataset: str, round_number: int) -> Optional[Dict]:
        """Resumo da estimativa da rodada (se já calculada)."""
        path = StoppingStatusReader.summary_path(self.validation_dir(dataset), round_number)
        if not path.exists():
            return None
        summary = pd.read_csv(path)
        groups = {}
        for group, rows in summary.groupby("grupo"):
            by_metric = rows.set_index("metrica")
            primary = by_metric.loc[PRIMARY_METRIC]
            groups[group] = {
                "n": int(primary["n"]),
                "kappa_fleiss": None if pd.isna(primary["kappa_fleiss"]) else float(primary["kappa_fleiss"]),
                "acordo_unanime": None if pd.isna(primary["acordo_unanime"]) else float(primary["acordo_unanime"]),
                "acordo_par_a_par": None if pd.isna(primary["acordo_par_a_par"]) else float(primary["acordo_par_a_par"]),
                "status": primary["status"],
                "metricas": {
                    m: {k: float(by_metric.loc[m, k]) for k in ("theta", "ic_inferior", "ic_superior", "moe")}
                    for m in (PRIMARY_METRIC, *SituationClassifier.SITUATIONS)
                },
            }
        stopped = all(g["status"] == StoppingStatusReader.STOP for g in groups.values())
        return {"rodada": round_number, "grupos": groups, "todos_pararam": stopped}

    # ------------------------------------------------------------- automação
    AUTO_NEXT_KEY = "proxima_automatica"
    AUTO_CLOSE_KEY = "fechamento_automatico"

    def _flag(self, key: str, default: bool) -> bool:
        value = self.store.get_option(key)
        return default if value is None else value == "1"

    def automation(self) -> Dict:
        """Escolha do administrador (banco) ou, se ainda não houver, o padrão da configuração."""
        return {
            self.AUTO_NEXT_KEY: self._flag(self.AUTO_NEXT_KEY, self.settings.auto_next_round),
            self.AUTO_CLOSE_KEY: self._flag(self.AUTO_CLOSE_KEY, self.settings.review_window_minutes > 0),
            "janela_revisao_minutos": self.settings.review_window_minutes,
        }

    def set_automation(self, auto_next: Optional[bool] = None, auto_close: Optional[bool] = None) -> Dict:
        for key, value in ((self.AUTO_NEXT_KEY, auto_next), (self.AUTO_CLOSE_KEY, auto_close)):
            if value is not None:
                self.store.set_option(key, "1" if value else "0")
        logger.info(f"Automação atualizada: {self.automation()}")
        return self.automation()

    def review_deadline(self, dataset: str) -> Optional[datetime]:
        """Fim da janela de revisão: última resposta + janela, só com rodada aberta e todos completos."""
        round_number = self.store.open_round(self.store_key(dataset))
        if round_number is None or not self.settings.review_window_minutes or not self.automation()[self.AUTO_CLOSE_KEY]:
            return None
        if any(p["faltam"] for p in self.progress(dataset, round_number).values()):
            return None
        last = self.store.last_answer_at(self.store_key(dataset), round_number)
        return datetime.fromisoformat(last) + timedelta(minutes=self.settings.review_window_minutes) if last else None

    # ------------------------------------------------------------- CSV de consenso
    def _sampled(self, dataset: str) -> Optional[SamplingLedger]:
        """Registro da amostragem, se a rodada 1 já foi sorteada."""
        ledger = SamplingLedger(self.validation_dir(dataset))
        if not ledger.exists():
            return None
        ledger.load()
        return ledger

    def consensus_status(self, dataset: str) -> Dict:
        return {
            "data": self.settings.experiments[dataset],
            "consenso_disponivel": self.consensus.exists(dataset, self.settings.experiments[dataset]),
            "consenso_bloqueado": self._sampled(dataset) is not None,
        }

    def upload_consensus(self, dataset: str, content: bytes) -> Dict:
        """Salva o CSV de consenso; depois do 1º sorteio, só aceita o MESMO arquivo (o registro confere o sha256)."""
        self._check_dataset(dataset)
        with self._lock:
            ledger = self._sampled(dataset)
            if ledger and ledger.data["source"]["sha256"] != self.consensus.sha256(content):
                raise RoundStateError(
                    "Este experimento já tem rodadas sorteadas com outro CSV. Para trocar os dados, "
                    "configure outra data (novo experimento) ou reinicie este dataset."
                )
            return self.consensus.save(dataset, self.settings.experiments[dataset], content)

    def status(self, dataset: str) -> Dict:
        self._check_dataset(dataset)
        current = self.store.current_round(self.store_key(dataset))
        if current is None:
            consensus = self.consensus_status(dataset)
            return {"dataset": dataset, "rodada": None, "estado": "sem_rodada",
                    "pode_iniciar": consensus["consenso_disponivel"], **consensus}

        round_number, state = current["rodada"], current["estado"]
        progress = self.progress(dataset, round_number)
        complete = all(p["faltam"] == 0 for p in progress.values())
        # Resultado da última rodada fechada (a atual, ou a anterior se a atual está aberta)
        closed = round_number if state == "fechada" else round_number - 1
        result = self.last_result(dataset, closed) if closed > 0 else None
        info = self.round_history(dataset)[-1]
        return {
            "dataset": dataset,
            "rodada": round_number,
            "estado": state,
            "aberta_em": info["aberta_em"],
            "concluida_em": info["concluida_em"],
            "fechada_em": info["fechada_em"],
            "duracao_segundos": info["duracao_segundos"],
            "progresso": progress,
            "pode_fechar": state == "aberta" and complete,
            "pode_iniciar": state == "fechada" and not (result and result["todos_pararam"]),
            "proxima_automatica": self.automation()[self.AUTO_NEXT_KEY],
            "fecha_automaticamente_em": self._iso(self.review_deadline(dataset)),
            "resultado": result,
            "planilha_disponivel": self.workbook_path(dataset).exists(),
            **self.consensus_status(dataset),
        }

    @staticmethod
    def _iso(moment: Optional[datetime]) -> Optional[str]:
        return moment.isoformat(timespec="seconds") if moment else None

    # ------------------------------------------------------------- painel
    @staticmethod
    def _seconds(start: Optional[str], end: Optional[str]) -> Optional[int]:
        if not start or not end:
            return None
        return int((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())

    def round_history(self, dataset: str) -> List[Dict]:
        """Rodadas com datas; duração = da abertura até o último avaliador terminar."""
        return [
            {**r, "duracao_segundos": self._seconds(r["aberta_em"], r["concluida_em"])}
            for r in self.store.rounds(self.store_key(dataset))
        ]

    # Sempre acumulado até a rodada (procedimento sequencial de Merlo et al.): rodadas nunca são lidas sozinhas
    SCOPE = "acumulado"

    def dashboard(self, dataset: str) -> Dict:
        """Painel de UM dataset: rodadas fechadas e métricas acumuladas até cada uma."""
        self._check_dataset(dataset)
        metrics = self.store.round_metrics(self.store_key(dataset))
        metrics = metrics[metrics["escopo"] == self.SCOPE].drop(columns=["dataset", "escopo"])
        return {
            "dataset": dataset,
            "rodadas": [r for r in self.round_history(dataset) if r["estado"] == "fechada"],
            "metricas": metrics.astype(object).where(metrics.notna(), None).to_dict(orient="records"),
        }

    def _save_round_metrics(self, dataset: str, round_number: int, cumulative: pd.DataFrame) -> None:
        key = self.store_key(dataset)
        self.store.save_round_metrics(key, round_number, self.SCOPE, cumulative)
        path = self.validation_dir(dataset) / StoppingStatusReader.OUTPUT_SUBDIR / "metricas_por_rodada.csv"
        self.store.round_metrics(key).to_csv(path, index=False)

    # ------------------------------------------------------------- aviso por email
    @staticmethod
    def _format_duration(seconds: Optional[int]) -> str:
        if seconds is None:
            return "—"
        hours, rest = divmod(seconds, 3600)
        return f"{hours}h {rest // 60:02d}min" if hours else f"{rest // 60}min"

    def _notify_round_closed(self, dataset: str, round_number: int) -> None:
        closed = next(r for r in self.round_history(dataset) if r["rodada"] == round_number)
        result = self.last_result(dataset, round_number) or {"grupos": {}, "todos_pararam": False}
        status = self.status(dataset)

        lines = [
            f"A rodada {round_number} de {dataset} ({self.settings.experiments[dataset]}) foi fechada.",
            f"Duração (abertura até o último avaliador terminar): {self._format_duration(closed['duracao_segundos'])}",
            f"Documentos na rodada: {closed['documentos']}",
            "",
            "Critério de parada (humano = referência, acumulado):",
        ]
        for group, g in result["grupos"].items():
            m = g["metricas"][PRIMARY_METRIC]
            lines.append(
                f"  Grupo {group}: {m['theta']:.1%} [{m['ic_inferior']:.1%} - {m['ic_superior']:.1%}], "
                f"MoE {m['moe']:.1%} (n={g['n']}) -> {g['status']}"
            )
        lines += ["", "Benchmark mislabeling (acumulado):"]
        for group, g in result["grupos"].items():
            m = g["metricas"]["benchmark_mislabeling"]
            lines.append(f"  Grupo {group}: {m['theta']:.1%} [{m['ic_inferior']:.1%} - {m['ic_superior']:.1%}]")

        lines.append("")
        if result["todos_pararam"]:
            lines.append(f"Todos os grupos atingiram o critério de parada: validação de {dataset} concluída.")
        elif status["estado"] == "aberta" and status["rodada"] == round_number + 1:
            total = next(iter(status["progresso"].values()))["total"]
            lines.append(f"A rodada {round_number + 1} já foi aberta ({total} documentos). Avise os avaliadores.")
        else:
            lines.append("A próxima rodada não foi aberta automaticamente: gere-a pelo painel /admin e avise os avaliadores.")

        self.notifier.send(f"[Validação humana] {dataset}: rodada {round_number} fechada", "\n".join(lines))

    # ------------------------------------------------------------- reinício
    BACKUP_DIR = "_backup"

    @staticmethod
    def _stamp() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def reset_dataset(self, dataset: str) -> str:
        """Zera a validação de UM dataset (os demais seguem intactos); retorna a pasta de backup."""
        self._check_dataset(dataset)
        with self._lock:
            return self._reset_experiment(dataset, self._stamp())

    def reset_all(self) -> List[str]:
        """Zera a validação de todos os experimentos; retorna as pastas de backup."""
        with self._lock:
            stamp = self._stamp()
            return [self._reset_experiment(dataset, stamp) for dataset in self.settings.experiments]

    def _reset_experiment(self, dataset: str, stamp: str) -> str:
        """Move arquivos e exporta o banco do experimento para `_backup/` e então o apaga."""
        key = self.store_key(dataset)
        date = self.settings.experiments[dataset]
        backup = HumanValidationPipeline.validation_root(self.settings.results_dir) / self.BACKUP_DIR / f"{dataset}_{date}_{stamp}"
        backup.mkdir(parents=True, exist_ok=True)

        for table, rows in self.store.export_experiment(key).items():
            rows.to_csv(backup / f"banco_{table}.csv", index=False)
        source = self.validation_dir(dataset)
        if source.exists():
            shutil.move(str(source), str(backup / "arquivos"))
        self.store.delete_experiment(key)

        logger.warning(f"{dataset}: validação reiniciada (backup em {backup})")
        return str(backup)

    # ------------------------------------------------------------- ciclo
    def close_due_rounds(self, now: Optional[datetime] = None) -> List[str]:
        """Fecha as rodadas cuja janela de revisão terminou; retorna os datasets fechados."""
        now = now or datetime.now()
        closed = []
        for dataset in self.settings.experiments:
            deadline = self.review_deadline(dataset)
            if deadline is None or now < deadline:
                continue
            try:
                self.close_round(dataset)
                closed.append(dataset)
            except RoundStateError:
                pass  # fechada pelo admin no meio do caminho
        return closed

    def start_round(self, dataset: str) -> Dict:
        """Sorteia (ou republica) a próxima rodada e a abre para os avaliadores."""
        with self._lock:
            return self._start_round(dataset)

    def _start_round(self, dataset: str) -> Dict:
        self._check_dataset(dataset)
        if self.store.open_round(self.store_key(dataset)) is not None:
            raise RoundStateError("Já existe uma rodada aberta para este dataset")
        if not self.consensus.exists(dataset, self.settings.experiments[dataset]):
            raise RoundStateError(f"Envie o CSV de consenso de {dataset} antes de abrir a primeira rodada")

        current = self.store.current_round(self.store_key(dataset))
        published = current["rodada"] if current else 0
        ledger = SamplingLedger(self.validation_dir(dataset))
        sampled = 0
        if ledger.exists():
            ledger.load()
            sampled = ledger.last_round

        pipeline = HumanValidationPipeline(self._sampling_config(dataset))
        # Rodada já sorteada mas não publicada (ex.: falha anterior): republica a mesma
        round_dir = pipeline.run(round_number=published + 1 if sampled > published else None)
        if round_dir is None:
            raise RoundStateError("Todos os grupos atingiram o critério de parada: nenhuma rodada nova")

        round_number = int(round_dir.name.split("_")[-1])
        documents = pd.read_csv(round_dir / HumanValidationPipeline.DOCUMENTS_FILE)
        self.store.publish_round(self.store_key(dataset), round_number, zip(documents["id_anonimo"], documents["texto"]))
        logger.success(f"{dataset}: rodada {round_number} aberta com {len(documents)} documentos")
        return self.status(dataset)

    def close_round(self, dataset: str) -> Dict:
        """Fecha a rodada se todos terminaram: estimação, planilha e (se configurado) abre a próxima."""
        with self._lock:
            return self._close_round(dataset)

    def _close_round(self, dataset: str) -> Dict:
        self._check_dataset(dataset)
        round_number = self.store.open_round(self.store_key(dataset))
        if round_number is None:
            raise RoundStateError("Não há rodada aberta para este dataset")

        missing = {e: p["faltam"] for e, p in self.progress(dataset, round_number).items() if p["faltam"]}
        if missing:
            raise RoundIncompleteError(f"Avaliadores com documentos pendentes: {missing}")

        # Trava antes de estimar (nenhuma edição durante a análise); reabre se a análise falhar
        self.store.close_round(self.store_key(dataset), round_number)
        try:
            estimation = HumanValidationEstimationPipeline(
                HumanValidationEstimationConfig(
                    dataset_name=dataset,
                    specific_date=self.settings.experiments[dataset],
                    results_dir=self.settings.results_dir,
                    evaluators=self.settings.evaluators,
                ),
                response_source=DatabaseResponseSource(self.store, self.store_key(dataset)),
            )
            summary = estimation.run()
            self._save_round_metrics(dataset, round_number, summary)
            self._write_workbook(dataset, round_number, summary)
        except Exception:
            self.store.reopen_round(self.store_key(dataset), round_number)
            raise
        logger.success(f"{dataset}: rodada {round_number} fechada")

        result = self.last_result(dataset, round_number)
        if self.automation()[self.AUTO_NEXT_KEY] and result and not result["todos_pararam"]:
            self._start_next_automatically(dataset)
        try:
            self._notify_round_closed(dataset, round_number)
        except Exception as e:
            logger.error(f"{dataset}: falha ao montar o aviso por email: {e}")
        return self.status(dataset)

    def _start_next_automatically(self, dataset: str) -> None:
        """Abre a próxima rodada; se falhar, a rodada fechada continua válida e o botão manual segue disponível."""
        try:
            self.start_round(dataset)
        except Exception as e:
            logger.error(f"{dataset}: não foi possível abrir a próxima rodada automaticamente: {e}")

    def _write_workbook(self, dataset: str, round_number: int, summary: pd.DataFrame) -> None:
        estimates = self.validation_dir(dataset) / StoppingStatusReader.OUTPUT_SUBDIR
        self.workbook_writer.write(
            self.workbook_path(dataset),
            responses=self.store.all_responses(self.store_key(dataset), round_number),
            documents=pd.read_csv(estimates / f"documentos_ate_rodada_{round_number:02d}.csv"),
            summary=summary,
            history=pd.read_csv(estimates / "historico.csv"),
        )
