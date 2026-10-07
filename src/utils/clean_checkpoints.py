from pathlib import Path

from loguru import logger


def clean_checkpoints(root_dir: str | Path, keep_suffixes: tuple[str, ...] = (".json",)) -> int:
    """Apaga de toda pasta checkpoint-* sob root_dir os arquivos fora de keep_suffixes (pesos, optimizer, scheduler...); retorna os bytes liberados."""
    freed = 0
    for checkpoint_dir in Path(root_dir).rglob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        for file in checkpoint_dir.rglob("*"):
            if file.is_file() and file.suffix not in keep_suffixes:
                freed += file.stat().st_size
                file.unlink()

    logger.info(f"Checkpoints limpos em {root_dir}: {freed / 1024**3:.2f} GB liberados")
    return freed


def main() -> None:
    # Configuração estática: pasta raiz a limpar (ex.: "data/results/agnews/2026-04-09_13-20-16/finetuning")
    # Não rode sobre um fine-tuning em andamento: o Trainer recarrega os pesos do melhor checkpoint no fim
    root_dir = "data/results"

    clean_checkpoints(root_dir)


if __name__ == "__main__":
    main()
