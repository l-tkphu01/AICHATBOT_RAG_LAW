from typing import Dict, Tuple
from app.utils.config_loader import Settings

class ModelRouter:
    """Layer 3: Model Routing"""

    def __init__(self, config: dict = None):
        if config is None:
            settings = Settings()
            self.config = settings.guardian_config.get("model_routing", {})
        else:
            self.config = config
            
        self.enabled = self.config.get("enabled", True)
        self.skip_conditions = self.config.get("skip_model_when", [])
        self.force_conditions = self.config.get("force_model_when", [])

    def _evaluate_condition(self, condition: str, features: Dict[str, float]) -> bool:
        try:
            for op in ['==', '>=', '<=', '>', '<']:
                if op in condition:
                    left, right = condition.split(op)
                    left = left.strip()
                    right = float(right.strip())
                    
                    left_val = features.get(left, 0.0)
                    
                    if op == '==': return left_val == right
                    if op == '>=': return left_val >= right
                    if op == '<=': return left_val <= right
                    if op == '>': return left_val > right
                    if op == '<': return left_val < right
            return False
        except Exception:
            return False

    def route(self, features: Dict[str, float]) -> Tuple[bool, str]:
        """
        Gives whether to use LLM intent model or not based on rules.
        Returns: (use_model, reasoning)
        """
        if not self.enabled:
            return True, "model_routing_disabled"
            
        # 1. Check skip conditions first (bypass model completely = safe/legal by default)
        for combined_cond in self.skip_conditions:
            sub_conds = combined_cond.split(' AND ')
            if all(self._evaluate_condition(sc, features) for sc in sub_conds):
                return False, f"skip_model: {combined_cond}"
                
        # 2. Check force model conditions (require model to verify)
        for combined_cond in self.force_conditions:
            sub_conds = combined_cond.split(' AND ')
            if all(self._evaluate_condition(sc, features) for sc in sub_conds):
                return True, f"force_model: {combined_cond}"
                
        # Default behavior if none matches
        return True, "default_to_model"
