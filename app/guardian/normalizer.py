import re
import unicodedata
from app.utils.config_loader import Settings

class Normalizer:
    """Layer 1: Normalization according to guardian_config"""
    
    def __init__(self, config: dict = None):
        if config is None:
            settings = Settings()
            config = settings.guardian_config.get("normalization", {})
            
        self.enabled = config.get("enabled", True)
        self.lowercase = config.get("lowercase", True)
        self.trim_whitespace = config.get("trim_whitespace", True)
        self.collapse_repeated_spaces = config.get("collapse_repeated_spaces", True)
        self.normalize_punctuation = config.get("normalize_punctuation", True)
        self.normalize_unicode_nfkc = config.get("normalize_unicode_nfkc", True)
        self.preserve_diacritics = config.get("preserve_diacritics", True)
        self.teencode_map = config.get("teencode_map", {})

    def normalize(self, text: str) -> str:
        if not self.enabled:
            return text
            
        if self.lowercase:
            text = text.lower()
            
        if self.normalize_unicode_nfkc:
            text = unicodedata.normalize('NFKC', text)
            
        if self.normalize_punctuation:
            # Basic punctuation normalization (can be expanded)
            text = text.replace('`', "'").replace('â€œ', '"').replace('â€', '"').replace('...', 'â€¦')
            
        if self.collapse_repeated_spaces:
            text = re.sub(r'\s+', ' ', text)
            
        if self.trim_whitespace:
            text = text.strip()
            
        # Teencode mapping mapping
        if self.teencode_map:
            words = text.split()
            normalized_words = [self.teencode_map.get(w, w) for w in words]
            text = " ".join(normalized_words)
            
        return text
