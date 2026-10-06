"""
Response Store - Persistência SQLite de rodadas, documentos, respostas e sessões
"""
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger

SCHEMA = """
CREATE TABLE IF NOT EXISTS rodadas (
    dataset TEXT NOT NULL,
    rodada INTEGER NOT NULL,
    estado TEXT NOT NULL CHECK (estado IN ('aberta', 'fechada')),
    aberta_em TEXT NOT NULL,
    fechada_em TEXT,
    PRIMARY KEY (dataset, rodada)
);
CREATE TABLE IF NOT EXISTS documentos (
    dataset TEXT NOT NULL,
    rodada INTEGER NOT NULL,
    posicao INTEGER NOT NULL,
    id_anonimo TEXT NOT NULL,
    texto TEXT NOT NULL,
    PRIMARY KEY (dataset, rodada, id_anonimo),
    UNIQUE (dataset, rodada, posicao)
);
CREATE TABLE IF NOT EXISTS respostas (
    dataset TEXT NOT NULL,
    rodada INTEGER NOT NULL,
    avaliador TEXT NOT NULL,
    id_anonimo TEXT NOT NULL,
    rotulo_escolhido TEXT NOT NULL,
    outro_rotulo_possivel TEXT NOT NULL,
    qual_outro_rotulo TEXT,
    atualizado_em TEXT NOT NULL,
    PRIMARY KEY (dataset, rodada, avaliador, id_anonimo)
);
CREATE TABLE IF NOT EXISTS sessoes (
    token TEXT PRIMARY KEY,
    usuario TEXT NOT NULL,
    papel TEXT NOT NULL,
    criada_em TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS configuracoes (
    chave TEXT PRIMARY KEY,
    valor TEXT NOT NULL
);
"""

ANSWER_FIELDS = ("rotulo_escolhido", "outro_rotulo_possivel", "qual_outro_rotulo")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ResponseStore:
    """
    Guarda o estado da interface em um arquivo SQLite.

    Responsabilidades:
    - Rodadas (aberta/fechada) e a lista cega de documentos de cada rodada
    - Respostas por avaliador, gravadas a cada envio (upsert)
    - Sessões de login (token -> usuário/papel)
    - Opções escolhidas pelo administrador na tela (ex.: automação)

    Nunca armazena gabarito, grupo ou classe: só id_anonimo e texto.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)
        logger.debug(f"ResponseStore inicializado: {self.db_path}")

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ opções
    def get_option(self, key: str) -> Optional[str]:
        with self._connect() as con:
            row = con.execute("SELECT valor FROM configuracoes WHERE chave = ?", (key,)).fetchone()
        return row["valor"] if row else None

    def set_option(self, key: str, value: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO configuracoes VALUES (?, ?) ON CONFLICT (chave) DO UPDATE SET valor = excluded.valor",
                (key, value),
            )

    # ------------------------------------------------------------------ sessões
    def create_session(self, user: str, role: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._connect() as con:
            con.execute("INSERT INTO sessoes VALUES (?, ?, ?, ?)", (token, user, role, _now()))
        return token

    def get_session(self, token: str) -> Optional[Tuple[str, str]]:
        with self._connect() as con:
            row = con.execute("SELECT usuario, papel FROM sessoes WHERE token = ?", (token,)).fetchone()
        return (row["usuario"], row["papel"]) if row else None

    # ------------------------------------------------------------------ rodadas
    def publish_round(self, dataset: str, round_number: int, documents: Iterable[Tuple[str, str]]) -> None:
        """Abre a rodada com os documentos (id_anonimo, texto) na ordem dada."""
        with self._connect() as con:
            con.execute("INSERT INTO rodadas VALUES (?, ?, 'aberta', ?, NULL)", (dataset, round_number, _now()))
            con.executemany(
                "INSERT INTO documentos VALUES (?, ?, ?, ?, ?)",
                [(dataset, round_number, pos, id_anon, text) for pos, (id_anon, text) in enumerate(documents, start=1)],
            )

    def close_round(self, dataset: str, round_number: int) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE rodadas SET estado = 'fechada', fechada_em = ? WHERE dataset = ? AND rodada = ?",
                (_now(), dataset, round_number),
            )

    def reopen_round(self, dataset: str, round_number: int) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE rodadas SET estado = 'aberta', fechada_em = NULL WHERE dataset = ? AND rodada = ?",
                (dataset, round_number),
            )

    def current_round(self, dataset: str) -> Optional[Dict]:
        """Última rodada do dataset (aberta ou fechada)."""
        with self._connect() as con:
            row = con.execute(
                "SELECT rodada, estado FROM rodadas WHERE dataset = ? ORDER BY rodada DESC LIMIT 1", (dataset,)
            ).fetchone()
        return dict(row) if row else None

    def open_round(self, dataset: str) -> Optional[int]:
        current = self.current_round(dataset)
        return current["rodada"] if current and current["estado"] == "aberta" else None

    # --------------------------------------------------------------- documentos
    def round_size(self, dataset: str, round_number: int) -> int:
        with self._connect() as con:
            return con.execute(
                "SELECT COUNT(*) FROM documentos WHERE dataset = ? AND rodada = ?", (dataset, round_number)
            ).fetchone()[0]

    def get_document(self, dataset: str, round_number: int, id_anonimo: str) -> Optional[Dict]:
        with self._connect() as con:
            row = con.execute(
                "SELECT posicao, id_anonimo, texto FROM documentos WHERE dataset = ? AND rodada = ? AND id_anonimo = ?",
                (dataset, round_number, id_anonimo),
            ).fetchone()
        return dict(row) if row else None

    def next_unanswered(self, dataset: str, round_number: int, evaluator: str) -> Optional[Dict]:
        """Primeiro documento, na ordem da rodada, ainda sem resposta do avaliador."""
        with self._connect() as con:
            row = con.execute(
                """SELECT d.posicao, d.id_anonimo, d.texto FROM documentos d
                   WHERE d.dataset = ? AND d.rodada = ? AND NOT EXISTS (
                       SELECT 1 FROM respostas r WHERE r.dataset = d.dataset AND r.rodada = d.rodada
                       AND r.id_anonimo = d.id_anonimo AND r.avaliador = ?)
                   ORDER BY d.posicao LIMIT 1""",
                (dataset, round_number, evaluator),
            ).fetchone()
        return dict(row) if row else None

    # ---------------------------------------------------------------- respostas
    def save_response(self, dataset: str, round_number: int, evaluator: str, id_anonimo: str, answer: Dict) -> None:
        values = tuple(answer.get(f) for f in ANSWER_FIELDS)
        with self._connect() as con:
            con.execute(
                f"""INSERT INTO respostas (dataset, rodada, avaliador, id_anonimo, {', '.join(ANSWER_FIELDS)}, atualizado_em)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (dataset, rodada, avaliador, id_anonimo) DO UPDATE SET
                   rotulo_escolhido = excluded.rotulo_escolhido,
                   outro_rotulo_possivel = excluded.outro_rotulo_possivel,
                   qual_outro_rotulo = excluded.qual_outro_rotulo,
                   atualizado_em = excluded.atualizado_em""",
                (dataset, round_number, evaluator, id_anonimo, *values, _now()),
            )

    def evaluator_responses(self, dataset: str, round_number: int, evaluator: str) -> List[Dict]:
        """Respostas do próprio avaliador, na ordem da rodada."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT d.posicao, r.id_anonimo, r.rotulo_escolhido, r.outro_rotulo_possivel,
                          r.qual_outro_rotulo, r.atualizado_em
                   FROM respostas r JOIN documentos d USING (dataset, rodada, id_anonimo)
                   WHERE r.dataset = ? AND r.rodada = ? AND r.avaliador = ? ORDER BY d.posicao""",
                (dataset, round_number, evaluator),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_response(self, dataset: str, round_number: int, evaluator: str, id_anonimo: str) -> Optional[Dict]:
        with self._connect() as con:
            row = con.execute(
                f"SELECT {', '.join(ANSWER_FIELDS)}, atualizado_em FROM respostas "
                "WHERE dataset = ? AND rodada = ? AND avaliador = ? AND id_anonimo = ?",
                (dataset, round_number, evaluator, id_anonimo),
            ).fetchone()
        return dict(row) if row else None

    def last_answer_at(self, dataset: str, round_number: int) -> Optional[str]:
        """Momento da resposta mais recente da rodada (ISO)."""
        with self._connect() as con:
            return con.execute(
                "SELECT MAX(atualizado_em) FROM respostas WHERE dataset = ? AND rodada = ?", (dataset, round_number)
            ).fetchone()[0]

    def progress(self, dataset: str, round_number: int, evaluators: List[str]) -> Dict[str, int]:
        """Nº de documentos respondidos por avaliador na rodada."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT avaliador, COUNT(*) AS n FROM respostas WHERE dataset = ? AND rodada = ? GROUP BY avaliador",
                (dataset, round_number),
            ).fetchall()
        counts = {r["avaliador"]: r["n"] for r in rows}
        return {e: counts.get(e, 0) for e in evaluators}

    def export_experiment(self, dataset: str) -> Dict[str, pd.DataFrame]:
        """Todas as linhas de um experimento, por tabela (para backup antes de reiniciar)."""
        with self._connect() as con:
            return {
                table: pd.read_sql_query(f"SELECT * FROM {table} WHERE dataset = ?", con, params=(dataset,))
                for table in ("rodadas", "documentos", "respostas")
            }

    def delete_experiment(self, dataset: str) -> None:
        """Apaga rodadas, documentos e respostas de um experimento."""
        with self._connect() as con:
            for table in ("respostas", "documentos", "rodadas"):
                con.execute(f"DELETE FROM {table} WHERE dataset = ?", (dataset,))

    def all_responses(self, dataset: str, up_to_round: int) -> pd.DataFrame:
        """Respostas de todas as rodadas até `up_to_round`, em formato longo."""
        with self._connect() as con:
            rows = con.execute(
                f"""SELECT r.rodada, d.posicao, r.avaliador, r.id_anonimo, {', '.join('r.' + f for f in ANSWER_FIELDS)},
                           r.atualizado_em
                    FROM respostas r JOIN documentos d USING (dataset, rodada, id_anonimo)
                    WHERE r.dataset = ? AND r.rodada <= ? ORDER BY r.rodada, d.posicao, r.avaliador""",
                (dataset, up_to_round),
            ).fetchall()
        columns = ["rodada", "posicao", "avaliador", "id_anonimo", *ANSWER_FIELDS, "atualizado_em"]
        return pd.DataFrame([dict(r) for r in rows], columns=columns)
