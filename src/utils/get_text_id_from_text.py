import hashlib

def get_text_id_from_text(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()