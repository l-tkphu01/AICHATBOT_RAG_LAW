import re
from typing import Dict, Any, Tuple, List, Pattern
from app.utils.config_loader import Settings
from app.guardian.normalizer import Normalizer
from app.guardian.hard_gate.feature_extractor import FeatureExtractor
from app.guardian.hard_gate.hard_gate import HardGate
from app.guardian.intent.model_router import ModelRouter
from app.guardian.intent.intent_classifier import IntentClassifier

class GuardianPipeline:
    """
    The main Guardian entry point. 
    Executes Normalization -> Feature Extraction -> Rule Gate -> Model Routing -> Full LLM Intent (if needed).
    """
    
    def __init__(self):
        self.settings = Settings()
        self.normalizer = Normalizer()
        self.feature_extractor = FeatureExtractor()
        self.hard_gate = HardGate()
        self.model_router = ModelRouter()
        self.intent_classifier = None  # Lazy load to avoid init API keys if not needed
        self.query_config = self.settings.query_processing.get("routing", {})
        self.legal_override_config = self.settings.guardian_config.get("legal_context_override", {})
        self._legal_info_patterns = self._compile_patterns(
            self.legal_override_config.get("informational_patterns", [])
        )
        self._legal_context_patterns = self._compile_patterns(
            self.legal_override_config.get("legal_context_patterns", [])
        )
        self._instructional_or_evasion_patterns = self._compile_patterns(
            self.legal_override_config.get("instructional_or_evasion_patterns", [])
        )
        self._neutralize_scores = self.legal_override_config.get(
            "neutralize_scores", ["banned_score", "harmful_score"]
        )
        
        # Load system messages
        self.system_messages = self.settings.prompts_config.get("prompts", {})

    def _get_classifier(self) -> IntentClassifier:
        if self.intent_classifier is None:
            self.intent_classifier = IntentClassifier()
        return self.intent_classifier

    def _compile_patterns(self, patterns: List[str]) -> List[Pattern[str]]:
        compiled: List[Pattern[str]] = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                continue
        return compiled

    def _matches_any_pattern(self, query: str, patterns: List[Pattern[str]]) -> bool:
        return any(pattern.search(query) for pattern in patterns)

    def _looks_legal_domain(self, query: str) -> bool:
        normalized = query.lower()
        fallback_keywords = self.settings.guardian_config.get('intent_fallback', {}).get('fallback_keywords', {})
        legal_signals = fallback_keywords.get('legal_domain', [
            'luật', 'nghị định', 'thông tư', 'điều', 'khoản', 'điểm', 'ly hôn',
            'hôn nhân', 'gia đình', 'thuế', 'khai thuế', 'hoàn thuế', 'hợp đồng',
            'lao động', 'doanh nghiệp', 'đất đai', 'thừa kế', 'khởi kiện', 'tranh chấp',
            'bồi thường', 'phạt', 'xử phạt', 'tòa án', 'hồ sơ', 'thủ tục'
        ])
        return any(keyword in normalized for keyword in legal_signals)

    def _is_legal_information_query(self, query: str) -> bool:
        has_legal_context = self._looks_legal_domain(query) or self._matches_any_pattern(
            query, self._legal_context_patterns
        )
        asks_legal_information = self._matches_any_pattern(query, self._legal_info_patterns)
        asks_instructional_or_evasion = self._matches_any_pattern(
            query, self._instructional_or_evasion_patterns
        )
        return has_legal_context and asks_legal_information and not asks_instructional_or_evasion

    def _should_apply_legal_context_override(self, query: str, features: Dict[str, float]) -> bool:
        if not self.legal_override_config.get("enabled", True):
            return False
        if features.get("injection_score", 0.0) >= 1.0:
            return False
        if features.get("malicious_score", 0.0) >= 1.0:
            return False
        return self._is_legal_information_query(query)

    def process_query(self, raw_query: str) -> Dict[str, Any]:
        """
        Process a query through the entire Guardian pipeline.
        Returns a dict containing action, next_step, status, reasoning, and optional response.
        """
        result = {
            "action": "continue_pipeline",
            "status": "safe",
            "reasoning": "",
            "features": {},
            "response": None,
            "next_step": "rewrite_then_retrieve",
            "query": raw_query
        }

        # 1. Normalization
        normalized_query = self.normalizer.normalize(raw_query)
        result["normalized_query"] = normalized_query
        
        # 2. Limit validations
        # Very simple validation from input_validation layer
        max_chars = self.settings.guardian_config.get("input_validation", {}).get("max_input_chars", 3000)
        if len(normalized_query) > max_chars:
            result.update({
                "action": "reject_or_refuse",
                "status": "unsafe",
                "reasoning": f"Query exceeds max chars: {max_chars}",
                "response": self.system_messages.get("input_too_long", "CÃ¢u há»i quÃ¡ dÃ i.")
            })
            return result

        # 3. Feature Extraction
        features = self.feature_extractor.extract_features(normalized_query)

        # Legal-context override: avoid false positives on legal analysis queries.
        if self._should_apply_legal_context_override(normalized_query, features):
            features = dict(features)
            for score_name in self._neutralize_scores:
                if score_name in features:
                    features[score_name] = 0.0

        result["features"] = features
        
        # 4. Hard Gate Rules
        is_blocked, action, reason = self.hard_gate.evaluate(features)
        if is_blocked:
            result.update({
                "action": "reject_or_refuse",
                "status": action,
                "reasoning": f"Blocked by Hard Gate Rule: {reason}",
                "response": self.system_messages.get("refuse_unsafe_query", "Xin lá»—i, tÃ´i khÃ´ng thá»ƒ há»— trá»£ cÃ¢u há»i nÃ y.")
            })
            return result
            
        # 5. Model Routing
        use_model, routing_reason = self.model_router.route(features)
        
        if not use_model:
            # Safe by rules, bypass LLM
            result["reasoning"] = f"Bypassed LLM Intent: {routing_reason}"
            return result
            
        # 6. LLM Intent Classification
        classifier = self._get_classifier()
        intent_data = classifier.classify(normalized_query, features)
        
        # Process LLM Intnet Result based on `query_config.yaml -> routing` mapping
        intent_category = (intent_data.get("intent") or intent_data.get("intent_label") or "uncertain").lower()
        confidence = intent_data.get("confidence", 0.0)

        if intent_category in {"safe", "good_intent"}:
            result["intent"] = intent_data
            result["action"] = "continue_pipeline"
            result["next_step"] = "rewrite_then_retrieve"
            result["reasoning"] = f"LLM Intent Classifier: {intent_data.get('reasoning', '')}"
            return result

        if intent_category == "uncertain" and self._is_legal_information_query(normalized_query):
            result["intent"] = {
                **intent_data,
                "intent": "safe",
                "confidence": max(confidence, 0.6),
                "reasoning": intent_data.get("reasoning", "Legal-domain fallback applied"),
            }
            result["action"] = "continue_pipeline"
            result["next_step"] = "rewrite_then_retrieve"
            result["reasoning"] = f"Legal-domain fallback: {intent_data.get('reasoning', '')}"
            return result
        
        # Mapping intent to router actions based on query_config.yaml logic
        # 'safe', 'unsafe', 'out_of_scope', 'clarify', 'good_intent', 'bad_intent', 'uncertain'
        routing_policy = self.query_config.get(intent_category, self.query_config.get("uncertain", {}))
        
        result["intent"] = intent_data
        result["action"] = routing_policy.get("action", "reject_or_refuse")
        result["next_step"] = routing_policy.get("next_step", None)
        
        if result["action"] in ["reject_or_refuse", "request_more_context"]:
            response_key = routing_policy.get("response_key")
            result["response"] = self.system_messages.get(response_key, "TÃ´i khÃ´ng thá»ƒ tráº£ lá»i cÃ¢u há»i nÃ y.")
            # Cập nhật lại status của result để báo hiệu hệ thống chặn
            result["status"] = intent_category
            
        result["reasoning"] = f"LLM Intent Classifier: {intent_data.get('reasoning', '')}"

        return result
