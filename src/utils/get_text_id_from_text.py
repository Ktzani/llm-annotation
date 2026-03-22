import hashlib

def get_text_id_from_text(text):
    return hashlib.md5(text.encode()).hexdigest()