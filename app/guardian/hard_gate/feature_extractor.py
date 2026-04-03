import re
from typing import Dict
from app.utils.config_loader import Settings

class FeatureExtractor:
    """Layer 1b: Regex -> Feature Extraction"""
    
    def __init__(self, config: dict = None):
        if config is None:
            settings = Settings()
            # fallback to empty dicts if config isn't loaded correctly
            self.config = settings.guardian_config.get("feature_extraction", {})
        else:
            self.config = config
            
        self.regex_patterns = self.config.get("regex_patterns", {})
        self.feature_building = self.config.get("feature_building", {})
        
        # Pre-compile regexes for performance
        self.compiled_patterns = {}
        for category, patterns in self.regex_patterns.items():
            combined_pattern = "|".join([f"(?:{p})" for p in patterns])
            self.compiled_patterns[category] = re.compile(combined_pattern, re.IGNORECASE)
            
        # Also precompile for the counts (need to count all hits)
        self.individual_patterns = {}
        for category, patterns in self.regex_patterns.items():
            self.individual_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract scores for all features defined in config."""
        features = {}
        
        for feature_name, rule in self.feature_building.items():
            # e.g., 'banned_score' -> extract prefix 'banned'
            category = feature_name.replace("_score", "")
            method = rule.get("method", "binary")
            
            if category not in self.regex_patterns:
                features[feature_name] = 0.0
                continue
                
            if method == "binary":
                hit = bool(self.compiled_patterns[category].search(text))
                val_if_hit = rule.get("value_if_hit", 1.0)
                features[feature_name] = val_if_hit if hit else 0.0
                
            elif method == "count_weighted":
                weight = rule.get("weight_per_hit", 0.3)
                max_score = rule.get("max_score", 1.0)
                
                # Count total matches across all patterns in category
                match_count = 0
                for pattern in self.individual_patterns[category]:
                    match_count += len(pattern.findall(text))
                    
                score = min(match_count * weight, max_score)
                features[feature_name] = score
                
            else:
                features[feature_name] = 0.0
                
        return features
