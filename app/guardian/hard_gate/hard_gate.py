from typing import Dict, Any, Tuple
from app.utils.config_loader import Settings

class HardGate:
    """Layer 2: Hard Gate - Evaluates feature scores against block conditions."""
    
    def __init__(self, config: dict = None):
        if config is None:
            settings = Settings()
            self.config = settings.guardian_config.get("hard_gate", {})
        else:
            self.config = config
            
        self.enabled = self.config.get("enabled", True)
        self.block_conditions = self.config.get("block_when", [])
        self.action = self.config.get("action", "unsafe")

    def _evaluate_condition(self, condition: str, features: Dict[str, float]) -> bool:
        """
        Evaluate a simple string condition like 'injection_score == 1.0', 
        or 'legal_anchor_score >= 0.9'.
        """
        try:
            # We explicitly define the allowed operators
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
            # Return False on parsing errors
            return False

    def evaluate(self, features: Dict[str, float]) -> Tuple[bool, str, str]:
        """
        Evaluate features.
        Returns: (is_blocked, action, reason)
        """
        if not self.enabled:
            return False, "continue", ""
            
        for condition in self.block_conditions:
            # Simple handle for ' AND ' if present in stop logic
            # Currently block_when are list of single checks evaluated as OR
            if ' AND ' in condition:
                sub_conds = condition.split(' AND ')
                if all(self._evaluate_condition(sc, features) for sc in sub_conds):
                    return True, self.action, condition
            else:
                if self._evaluate_condition(condition, features):
                    return True, self.action, condition
                    
        return False, "continue", ""
