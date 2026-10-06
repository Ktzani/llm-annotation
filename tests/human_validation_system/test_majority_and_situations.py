"""
Maioria entre os três avaliadores e classificação nas quatro situações (com fronteiras).
"""
import pandas as pd
import pytest

from src.systems.human_validation_system.estimation.human_label_aggregator import HumanLabelAggregator
from src.systems.human_validation_system.estimation.situation_classifier import SituationClassifier
from tests.human_validation_system.conftest import UNDECIDABLE

N = 3


def build(cases):
    """cases: (id, referência, consolidado LLM, [(escolhido, outro, qual) x avaliadores])."""
    key, responses = [], []
    for doc_id, ref, llm, answers in cases:
        key.append({"id_anonimo": doc_id, "grupo": "A", "classe_referencia": ref, "rotulo_referencia": 0,
                    "rotulo_consolidado": llm})
        for i, (chosen, other, which) in enumerate(answers, start=1):
            responses.append({"id_anonimo": doc_id, "avaliador": f"av{i}", "rotulo_escolhido": chosen,
                              "outro_rotulo_possivel": other, "qual_outro_rotulo": which})
    responses = pd.DataFrame(responses)
    docs = HumanLabelAggregator(N).aggregate(responses, pd.DataFrame(key))
    return SituationClassifier(N, UNDECIDABLE).classify(responses, docs).set_index("id_anonimo")


NO = ("não", None)


# ------------------------------------------------------------------ maioria
@pytest.mark.parametrize("chosen, expected", [
    (["romance", "romance", "romance"], "romance"),
    (["romance", "romance", "history"], "romance"),
    (["romance", "history", "poetry"], None),
    ([UNDECIDABLE, UNDECIDABLE, "romance"], UNDECIDABLE),
    ([UNDECIDABLE, "romance", "history"], None),
])
def test_majority(chosen, expected):
    docs = build([("d", "history", "romance", [(c, *NO) for c in chosen])])
    value = docs.loc["d", "rotulo_humano"]
    assert (pd.isna(value) and expected is None) or value == expected


def test_incomplete_document_has_no_situation():
    docs = build([("d", "history", "romance", [("romance", *NO), ("romance", *NO)])])
    assert not docs.loc["d", "completo"]
    assert pd.isna(docs.loc["d", "situacao"])
    assert pd.isna(docs.loc["d", "acerto_referencia"])


def test_three_way_divergence_counts_as_not_confirmed():
    docs = build([("d", "history", "romance", [("romance", *NO), ("history", *NO), ("poetry", *NO)])])
    assert docs.loc["d", "acerto_referencia"] == 0
    assert docs.loc["d", "acerto_llm"] == 0


# ------------------------------------------------------------------ desfechos
@pytest.mark.parametrize("answers, expected", [
    # convergência para a referência / para outra classe
    ([("history", *NO)] * 3, "benchmark_correct"),
    ([("romance", *NO)] * 3, "benchmark_mislabeling"),
    ([("poetry", *NO), ("poetry", *NO), ("romance", *NO)], "benchmark_mislabeling"),
    # fronteira da ambiguidade: apoio do lado oposto = 2 (ambíguo) vs 1 (não)
    ([("romance", "sim", "history"), ("romance", "sim", "history"), ("romance", *NO)], "genuine_ambiguity"),
    ([("romance", "sim", "history"), ("romance", *NO), ("romance", *NO)], "benchmark_mislabeling"),
    ([("romance", *NO), ("romance", "sim", "history"), ("history", *NO)], "genuine_ambiguity"),
    ([("romance", *NO), ("romance", *NO), ("history", *NO)], "benchmark_mislabeling"),
    ([("history", "sim", "romance"), ("history", *NO), ("romance", *NO)], "genuine_ambiguity"),
    # alternativa fora do conflito não gera ambiguidade
    ([("romance", "sim", "poetry"), ("romance", "sim", "poetry"), ("romance", *NO)], "benchmark_mislabeling"),
    # avaliadores não convergem
    ([("romance", *NO), ("history", *NO), ("poetry", *NO)], "genuine_ambiguity"),
    # não é possível decidir
    ([(UNDECIDABLE, *NO), (UNDECIDABLE, *NO), ("romance", *NO)], "insufficient_information"),
    ([(UNDECIDABLE, *NO), ("romance", *NO), ("romance", *NO)], "benchmark_mislabeling"),
])
def test_situations_conflict_group(answers, expected):
    docs = build([("d", "history", "romance", answers)])
    assert docs.loc["d", "situacao"] == expected


@pytest.mark.parametrize("answers, expected", [
    ([("poetry", *NO)] * 3, "benchmark_correct"),
    ([("poetry", "sim", "children"), ("poetry", "sim", "children"), ("poetry", *NO)], "genuine_ambiguity"),
    ([("poetry", "sim", "children"), ("poetry", "sim", "romance"), ("poetry", *NO)], "benchmark_correct"),
    ([("poetry", *NO), ("poetry", *NO), (UNDECIDABLE, *NO)], "benchmark_correct"),
])
def test_situations_control_group(answers, expected):
    """Grupo C: referência = LLM; ambiguidade exige a MESMA alternativa com apoio da maioria."""
    docs = build([("d", "poetry", "poetry", answers)])
    assert docs.loc["d", "situacao"] == expected


def test_situation_indicators_are_exclusive():
    docs = build([
        ("a", "history", "romance", [("history", *NO)] * 3),
        ("b", "history", "romance", [("romance", *NO)] * 3),
        ("c", "history", "romance", [("romance", *NO), ("history", *NO), ("poetry", *NO)]),
        ("d", "history", "romance", [(UNDECIDABLE, *NO)] * 3),
    ])
    assert (docs[list(SituationClassifier.SITUATIONS)].sum(axis=1) == 1).all()
