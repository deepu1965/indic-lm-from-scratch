import unicodedata
import re

def normalize_for_match(text: str) -> str:
    """
    Conservatively normalize text for exact-match comparison.
    - Unicode normalization (NFC, consistent with project)
    - Collapse repeated whitespace and strip leading/trailing whitespace
    - Does NOT alter meaningful characters, punctuation, or Devanagari script.
    """
    if not isinstance(text, str):
        return ""
    
    # NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    return text.strip()
