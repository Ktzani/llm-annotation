"""
Pipeline de anotação em 2 fases (filtro de classes + LLM zero-shot).

Por fold externo (auto-descoberto no HF Hub, como no fine-tuning):
  1. Carrega o conjunto de treino do fold.
  2. Divide o treino em pedaços; separa o held-out (o LLM anota; o classificador
     treina no resto). Sem vazamento: o classificador nunca vê o held-out.
  3. Fase 1: treina o filtro no fit part; prevê probabilidades no held-out.
  4. Mede recall@k (teto de acurácia da Fase 2) e deriva as top-k candidatas.
  5. Fase 2: o LLM anota o held-out zero-shot, vendo só as k candidatas.
  6. Avalia contra o ground-truth e persiste os artefatos por fold.

Com `filter_active=False`, anota o MESMO held-out com TODAS as classes — baseline
comparável sobre exatamente os mesmos textos.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from src.api.schemas.annotation_experiment.dataset import DatasetConfig
from src.api.services.prompt_factory import get_prompt_template
from src.utils.data_loader import load_hf_dataset_as_dataframe
from src.utils.get_text_id_from_text import get_text_id_from_text
from src.systems.llm_annotation_system.pipeline import AnnotationConfig
from src.systems.llm_annotation_system.annotation.llm_annotator import LLMAnnotator
from src.systems.llm_annotation_system.core.evaluate_model_metrics import evaluate_model_metrics

from src.systems.class_filter_system.classifiers.factory import get_class_filter
from src.systems.class_filter_system.folds.inner_split import inner_split
from src.systems.class_filter_system.validation.recall_at_k import recall_at_k_sweep


class TwoPhaseAnnotationPipeline:
    """Orquestra a anotação em 2 fases por fold, respeitando a divisão existente."""

    def __init__(self, config: AnnotationConfig):
        """
        Args:
            config: objeto AnnotationConfig (o mesmo construído por run_annotation),
                já com `class_filter` populado a partir do experimento.
        """
        self.config = config
        self.cf = config.class_filter
        self.prompt_template = get_prompt_template(config.prompt_type, config.custom_prompt)
        logger.success("✓ Setup 2 fases completo")

    # ------------------------------------------------------------------
    # Descoberta de folds (mesma convenção do fine-tuning)
    # ------------------------------------------------------------------
    def _load_train_fold(self, fold: int) -> Optional[pd.DataFrame]:
        """Carrega train_fold_{fold}.parquet como DataFrame (text, label). None se não existir.

        Respeita `sample_size`/`random_state` do `dataset_config` do experimento —
        útil para limitar o tamanho do fold em smoke tests (o held-out é uma fração
        desse total).
        """
        hf_file = self.cf.train_fold_pattern.format(fold=fold)
        fold_config = DatasetConfig(
            hf_file=hf_file,
            sample_size=self.config.dataset_config.sample_size,
            random_state=self.config.dataset_config.random_state,
        )
        try:
            df, _categories = load_hf_dataset_as_dataframe(
                dataset_name=self.config.dataset_name,
                cache_dir=self.config.cache_dir,
                dataset_global_config=fold_config,
            )
            return df
        except Exception as e:
            logger.info(f"Fold {fold} indisponível ({hf_file}): parando descoberta. [{e}]")
            return None

    # ------------------------------------------------------------------
    # Anotação de um held-out (Fase 2), com ou sem filtro
    # ------------------------------------------------------------------
    async def _annotate_holdout(
        self,
        df_holdout: pd.DataFrame,
        categories: list,
        candidates_by_text_id: Optional[dict],
        out_dir: Path,
        checkpoint_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # O checkpoint (intermediate.csv) vive num caminho ESTÁVEL (fora do run
        # timestampado), keyed por k/fold/variante — assim um re-run retoma de
        # onde parou, pulando os text_ids já anotados. Os artefatos finais
        # (annotations.csv, model_metrics.csv) vão para `out_dir` (o run atual).
        annotator = LLMAnnotator(
            dataset_name=self.config.dataset_name,
            models=self.config.models,
            categories=categories,
            cache_dir=self.config.cache_dir,
            results_dir=str(checkpoint_dir),
            prompt_template=self.prompt_template,
            use_cache=self.config.use_cache,
            use_alternative_params=self.config.use_alternative_params,
            keep_alive=self.config.keep_alive,
            candidates_by_text_id=candidates_by_text_id,
        )

        texts = df_holdout["text"].tolist()
        df_ann = await annotator.annotate_dataset(
            texts=texts,
            num_repetitions=self.config.num_repetitions,
            intermediate=self.config.intermediate,
            model_strategy=self.config.model_strategy,
            rep_strategy=self.config.rep_strategy,
            max_concurrent_texts=self.config.max_concurrent_texts,
        )
        df_ann = df_ann.drop_duplicates(subset=["text_id"])

        # Mantém só o held-out atual: se o checkpoint estável acumulou text_ids de
        # um held-out anterior (ex.: random_state diferente), eles são descartados.
        holdout_ids = set(df_holdout["text_id"])
        df_ann = df_ann[df_ann["text_id"].isin(holdout_ids)]

        df_gt = df_holdout[["text_id", "label"]].rename(columns={"label": "ground_truth"})
        df_ann = df_ann.merge(df_gt, on="text_id", how="left")

        df_ann.to_csv(out_dir / "annotations.csv", index=False)
        evaluate_model_metrics(
            df_ann,
            models=annotator.models,
            ground_truth_col="ground_truth",
            output_dir=out_dir,
        )
        logger.success(f"✓ Anotações + métricas em: {out_dir}")

    # ------------------------------------------------------------------
    # Fase 1 (filtro) + orquestração por fold
    # ------------------------------------------------------------------
    async def run(self, run_baseline: bool = False) -> Path:
        """
        Executa a anotação em 2 fases para todos os folds externos.

        Args:
            run_baseline: se True, também anota o MESMO held-out de cada fold com
                todas as classes (baseline comparável, mesmos textos).
        """
        two_phase_root = (
            Path(self.config.results_dir) / self.config.dataset_name / "two_phase"
        )
        base_dir = two_phase_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Raiz ESTÁVEL dos checkpoints (fora do run timestampado), keyed por k —
        # permite retomar de onde parou entre runs. Trocar k → checkpoint novo,
        # evitando reusar anotações do k anterior.
        checkpoint_root = two_phase_root / "_checkpoints" / f"k{self.cf.k}"

        recall_frames = []
        fold = 0
        while True:
            df_train = self._load_train_fold(fold)
            if df_train is None:
                break

            logger.info("=" * 60)
            logger.info(f"FOLD {fold} | treino: {len(df_train)} textos")

            df_train = df_train.copy()
            df_train["text"] = df_train["text"].astype(str)
            df_train["text_id"] = df_train["text"].apply(get_text_id_from_text)

            # Alguns datasets (ex.: dblp) têm textos DUPLICADOS. Dedup por text_id
            # ANTES do split interno resolve dois problemas:
            #   (1) held-out com text_id repetido colapsaria `candidates_by_text_id`
            #       (dict keyed por text_id), quebrando o assert de cobertura; e
            #   (2) vazamento — a mesma instância cair em fit E held-out faria o LR
            #       treinar no texto que depois prevê.
            before = len(df_train)
            df_train = df_train.drop_duplicates(subset="text_id").reset_index(drop=True)
            if before != len(df_train):
                logger.warning(
                    f"FOLD {fold}: removidas {before - len(df_train)} duplicatas de "
                    f"text_id antes do split (evita colapso de candidatas e vazamento)"
                )

            categories = sorted(df_train["label"].unique().tolist())

            # Split interno: fit part x held-out
            df_fit, df_holdout = inner_split(
                df_train,
                n_inner_folds=self.cf.n_inner_folds,
                holdout_fold_index=self.cf.holdout_fold_index,
                random_state=self.cf.random_state,
                label_column="label",
            )
            df_holdout = df_holdout.reset_index(drop=True)

            # Fase 1: treina o filtro no fit part, prevê no held-out
            filt = get_class_filter(
                self.cf.method,
                random_state=self.cf.random_state,
                **self.cf.params,
            )
            filt.fit(df_fit["text"].tolist(), df_fit["label"].tolist())
            proba = filt.predict_proba(df_holdout["text"].tolist())

            # recall@k (teto) — no MESMO held-out e com o MESMO proba das candidatas
            df_recall = recall_at_k_sweep(
                y_true=df_holdout["label"].tolist(),
                proba=proba,
                classes=filt.classes_,
                k_values=self.cf.k_sweep,
            )
            df_recall.insert(0, "fold", fold)
            recall_frames.append(df_recall)
            fold_dir = base_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            df_recall.to_csv(fold_dir / "recall_at_k.csv", index=False)
            logger.info(f"recall@k (fold {fold}):\n{df_recall.to_string(index=False)}")

            # Candidatas top-k (ordem canônica) -> mapa text_id -> [classes]
            topk = filt.topk(df_holdout["text"].tolist(), k=self.cf.k)
            candidates_by_text_id = {
                tid: cands
                for tid, cands in zip(df_holdout["text_id"].tolist(), topk)
            }
            assert len(candidates_by_text_id) == len(df_holdout), "cobertura de candidatas incompleta"

            with open(fold_dir / "filter_report.json", "w", encoding="utf-8") as f:
                json.dump({
                    "fold": fold,
                    "method": self.cf.method,
                    "k": self.cf.k,
                    "n_classes": len(filt.classes_),
                    "classes": filt.classes_,
                    "n_fit": len(df_fit),
                    "n_holdout": len(df_holdout),
                    "recall_at_k": df_recall.drop(columns=["fold"]).to_dict("records"),
                }, f, indent=2, ensure_ascii=False)

            # Fase 2: LLM anota o held-out com espaço reduzido
            await self._annotate_holdout(
                df_holdout,
                categories,
                candidates_by_text_id,
                out_dir=fold_dir / "filtered",
                checkpoint_dir=checkpoint_root / f"fold_{fold}" / "filtered",
            )

            # Baseline comparável (mesmos textos, todas as classes)
            if run_baseline:
                await self._annotate_holdout(
                    df_holdout,
                    categories,
                    None,
                    out_dir=fold_dir / "baseline",
                    checkpoint_dir=checkpoint_root / f"fold_{fold}" / "baseline",
                )

            fold += 1

        if fold == 0:
            logger.error(
                f"Nenhum fold encontrado para '{self.config.dataset_name}' "
                f"(padrão: {self.cf.train_fold_pattern})."
            )
            return base_dir

        # Agregação do recall@k entre folds
        df_all = pd.concat(recall_frames, ignore_index=True)
        df_all.to_csv(base_dir / "recall_at_k_all_folds.csv", index=False)
        df_agg = (
            df_all.groupby("k")["recall_at_k"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        df_agg.to_csv(base_dir / "recall_at_k_aggregated.csv", index=False)
        logger.success(f"✓ Pipeline 2 fases finalizado em {fold} folds. Artefatos: {base_dir}")
        logger.info(f"recall@k agregado:\n{df_agg.to_string(index=False)}")

        return base_dir
