"""
Limpeza só de exibição: a descrição duplicada do books aparece uma vez para o avaliador.
"""
import pytest

from src.systems.human_validation_system.interface.api.services.display_text_cleaner import DisplayTextCleaner

DESC = ('"My fellow monsters," said Moreau. "No longer will we hide in the shadows." '
        "A year ago, Boy, the son of Frankenstein's monster, had never even met a human.")


@pytest.fixture
def cleaner():
    return DisplayTextCleaner()


def test_duplicated_description_is_shown_once_with_original_case(cleaner):
    text = f"man made boy (#2) . {DESC.lower()} {DESC}"
    assert cleaner.clean(text) == f"man made boy (#2) . {DESC}"


@pytest.mark.parametrize("text", [
    f"man made boy (#2) . {DESC}",                                   # sem duplicata
    f"man made boy (#2) . {DESC.lower()} {DESC[:60]}",               # segunda cópia incompleta
    f"man made boy (#2) . {DESC.lower()} outra coisa {DESC}",       # trecho diferente no meio
    "deep learning for parsing. we propose a new parser " * 3,      # sem o separador ' . ' de título
])
def test_other_texts_are_unchanged(cleaner, text):
    assert cleaner.clean(text) == text


def test_cleaning_never_drops_content(cleaner):
    text = f"title . {DESC.lower()}   {DESC}"
    title, _, body = cleaner.clean(text).partition(" . ")
    assert " ".join(f"{title} . {body.lower()} {body}".split()) == " ".join(text.split())
