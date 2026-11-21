"""
Simulação de Monte Carlo da pseudo-contagem k na variância do estimador estratificado.

Reproduz o procedimento sequencial da validação humana (rodada inicial, incrementos sem
reposição, parada com MoE <= ε após o n mínimo) sobre os estratos reais de cada grupo e mede,
para cada k, o n até parar e a cobertura real do IC de 95% (a nominal é 95%).
Usa as mesmas classes da interface (ProportionalAllocator e StratifiedEstimator).

Uso:
    poetry run python -m scripts.simulate_pseudo_count
Configuração: constantes abaixo. Saída: data/validacao_humana/simulacao/pseudo_contagem.csv
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from src.config.human_validation import (
    CONFIDENCE_LEVEL,
    INCREMENT_SIZE,
    INITIAL_ROUND_SIZE,
    MIN_PER_CLASS_FIRST_ROUND,
    MIN_SAMPLE_FOR_STOPPING,
    MOE_THRESHOLD,
)
from src.systems.human_validation_system.estimation.stratified_estimator import StratifiedEstimator
from src.systems.human_validation_system.sampling.proportional_allocator import ProportionalAllocator

# Experimentos cujos estratos (N_h por classe, sem os exemplos do guia) são usados
EXPERIMENTS = {"books": "2026-09-13_07-19-54", "dblp": "2026-09-13_18-46-30"}
VALIDATION_ROOT = Path("data/validacao_humana")
OUTPUT_PATH = VALIDATION_ROOT / "simulacao" / "pseudo_contagem.csv"

PSEUDO_COUNTS = [0, 0.25, 0.5, 1]  # 0 = Wald original (Merlo et al.)
TRUE_THETAS = [0.10, 0.30, 0.50, 0.70, 0.90, 0.96, 0.98]  # θ médio do grupo
RATE_CONCENTRATION = 20  # taxas por classe ~ Beta em torno de θ (heterogeneidade entre classes)
REPETITIONS = 500
SIMULATION_SEED = 0
WORKERS = max(1, (os.cpu_count() or 2) - 2)


class SequentialProcedureSimulator:
    """
    Simula o procedimento sequencial estratificado num grupo de população finita.

    Responsabilidades:
    - Gerar a população 0/1 por classe a partir das taxas de cada classe
    - Sortear rodadas sem reposição com a alocação proporcional da interface
    - Parar como a interface (n >= mínimo e MoE <= ε, ou população esgotada)
    - Medir n final, MoE final e se o IC cobre o θ real da população
    """

    def __init__(self, sizes: Dict[int, int], pseudo_count: float, rng: np.random.Generator):
        self.sizes = sizes
        self.rng = rng
        self.allocator = ProportionalAllocator()
        self.estimator = StratifiedEstimator(CONFIDENCE_LEVEL, pseudo_count)

    def _population(self, rates: Dict[int, float]) -> Dict[int, np.ndarray]:
        """Ordem aleatória já embaralhada: tomar os primeiros = sortear sem reposição."""
        return {h: self.rng.permutation(self.rng.random(n) < rates[h]) for h, n in self.sizes.items()}

    def run_once(self, rates: Dict[int, float]) -> Dict:
        population = self._population(rates)
        total = sum(self.sizes.values())
        true_theta = sum(population[h].mean() * self.sizes[h] for h in self.sizes) / total
        taken = {h: 0 for h in self.sizes}
        round_number = 0
        while True:
            round_number += 1
            first = round_number == 1
            available = {h: self.sizes[h] - taken[h] for h in self.sizes}
            allocation = self.allocator.allocate(
                INITIAL_ROUND_SIZE if first else INCREMENT_SIZE, self.sizes, available, taken,
                MIN_PER_CLASS_FIRST_ROUND if first else 0,
            )
            for h, n in allocation.items():
                taken[h] += n
            values = pd.DataFrame({
                "estrato": np.repeat(list(taken), list(taken.values())),
                "valor": np.concatenate([population[h][:taken[h]] for h in taken]),
            })
            summary, _ = self.estimator.estimate(values, self.sizes, "simulação")
            exhausted = sum(taken.values()) >= total
            if (summary["n"] >= MIN_SAMPLE_FOR_STOPPING and summary["moe"] <= MOE_THRESHOLD) or exhausted:
                break
        covered = summary["ic_inferior"] <= true_theta <= summary["ic_superior"]
        return {"n": summary["n"], "moe": summary["moe"], "rodadas": round_number, "cobre": covered}


def simulate_scenario(scenario: Dict) -> Dict:
    """Roda as repetições de um cenário (função de módulo para rodar em outro processo)."""
    logger.remove()
    simulator = SequentialProcedureSimulator(scenario["sizes"], scenario["k"], np.random.default_rng(scenario["seed"]))
    runs = pd.DataFrame([simulator.run_once(scenario["rates"]) for _ in range(REPETITIONS)])
    return {
        "dataset": scenario["dataset"], "grupo": scenario["grupo"], "N": sum(scenario["sizes"].values()),
        "theta_alvo": scenario["theta"], "k": scenario["k"],
        "n_medio": runs["n"].mean(), "n_p10": runs["n"].quantile(0.1), "n_p90": runs["n"].quantile(0.9),
        "rodadas_medias": runs["rodadas"].mean(), "moe_media": runs["moe"].mean(), "cobertura": runs["cobre"].mean(),
    }


class PseudoCountStudy:
    """
    Varre datasets, grupos, θ e k e resume cobertura e custo.

    Responsabilidades:
    - Ler os N_h reais de cada grupo no amostragem.json do experimento
    - Gerar taxas por classe heterogêneas em torno de cada θ (mesmas taxas para todos os k)
    - Rodar os cenários em paralelo, cada um com semente própria (resultado independe da ordem)
    """

    def __init__(self):
        self.seeds = np.random.SeedSequence(SIMULATION_SEED)
        self.rng = np.random.default_rng(self.seeds.spawn(1)[0])

    @staticmethod
    def strata_sizes(dataset: str, date: str) -> Dict[str, Dict[int, int]]:
        data = json.loads((VALIDATION_ROOT / dataset / date / "amostragem.json").read_text(encoding="utf-8"))
        return {g: {int(h): n for h, n in s.items()} for g, s in data["strata_sizes"].items()}

    def class_rates(self, sizes: Dict[int, int], theta: float) -> Dict[int, float]:
        a, b = theta * RATE_CONCENTRATION, (1 - theta) * RATE_CONCENTRATION
        return {h: float(self.rng.beta(a, b)) for h in sizes}

    def scenarios(self) -> List[Dict]:
        scenarios = []
        for dataset, date in EXPERIMENTS.items():
            for group, sizes in self.strata_sizes(dataset, date).items():
                for theta in TRUE_THETAS:
                    rates = self.class_rates(sizes, theta)
                    for k in PSEUDO_COUNTS:
                        scenarios.append({"dataset": dataset, "grupo": group, "sizes": sizes, "theta": theta,
                                          "rates": rates, "k": k})
        for scenario, seed in zip(scenarios, self.seeds.spawn(len(scenarios))):
            scenario["seed"] = seed
        return scenarios

    def run(self) -> pd.DataFrame:
        scenarios = self.scenarios()
        logger.info(f"{len(scenarios)} cenários em até {WORKERS} processos")
        rows = []
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for row in pool.map(simulate_scenario, scenarios):
                rows.append(row)
                logger.info(f"[{len(rows)}/{len(scenarios)}] {row['dataset']} {row['grupo']} θ={row['theta_alvo']:.2f} "
                            f"k={row['k']}: n={row['n_medio']:.0f} cobertura={row['cobertura']:.1%}")
        return pd.DataFrame(rows)


def main():
    logger.info(
        f"Simulação: rodada inicial {INITIAL_ROUND_SIZE}, incrementos de {INCREMENT_SIZE}, ε={MOE_THRESHOLD}, "
        f"{REPETITIONS} repetições por cenário"
    )
    result = PseudoCountStudy().run()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    logger.success(f"Resultado salvo em {OUTPUT_PATH}")
    table = result.pivot_table(index=["dataset", "grupo", "theta_alvo"], columns="k", values=["cobertura", "n_medio"])
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
