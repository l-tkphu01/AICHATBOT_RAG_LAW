import re
import random
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
        legal_signals = fallback_keywords.get('legal_domain', [])
        return any(keyword in normalized for keyword in legal_signals)

    def _is_legal_information_query(self, query: str) -> bool:
        has_legal_context = self._looks_legal_domain(query)
        return has_legal_context

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
        input_settings = self.settings.guardian_config.get("input_validation", {})
        max_chars = input_settings.get("max_input_chars", 2000)
        max_lines = input_settings.get("max_input_lines", 30)
        summarize_threshold = input_settings.get("summarize_threshold", 1000)
        warning_threshold = input_settings.get("warning_threshold", 1800)
        hard_limit_action = input_settings.get("hard_limit_action", "reject")
        reject_message_key = input_settings.get("reject_message_key", "input_too_long")
        
        reject_message = self.system_messages.get(reject_message_key, "Câu hỏi quá dài hoặc chứa quá nhiều dòng trống.")
        
        # Check max lines
        # Sửa lỗi: raw_query đếm dòng chính xác hơn vì normalized_query có thể đã bị collapse_repeated_spaces
        line_count = len(raw_query.split("\n"))
        if line_count > max_lines:
            if hard_limit_action == "reject":
                result.update({
                    "action": "reject_or_refuse",
                    "status": "unsafe",
                    "reasoning": f"Query exceeds max lines: {max_lines} (got {line_count})",
                    "response": reject_message
                })
                return result
                
        # Check max chars
        if len(normalized_query) > max_chars:
            if hard_limit_action == "reject":
                result.update({
                    "action": "reject_or_refuse",
                    "status": "unsafe",
                    "reasoning": f"Query exceeds max chars: {max_chars}",
                    "response": reject_message
                })
                return result

        # Check soft thresholds
        if len(normalized_query) > warning_threshold:
            result["warning"] = f"Query length is near the hard limit ({len(normalized_query)} chars)."
        
        if len(normalized_query) > summarize_threshold:
            result["action"] = input_settings.get("summarize_action", "summarize_then_classify")
            result["reasoning"] = f"Query length ({len(normalized_query)}) exceeds summarize threshold ({summarize_threshold})."
            # NOTE: If "summarize_then_classify" action is flagged, the main pipeline or intent classifier 
            # should handle it by calling the llm to summarize it using "guard_input_summarizer" prompt.
            # We pass the action through to signal downstream processing.

        # 3. Feature Extraction
        features = self.feature_extractor.extract_features(normalized_query)

        result["features"] = features
        
        # 4. Hard Gate Rules
        is_blocked, action, reason = self.hard_gate.evaluate(features)
        if is_blocked:
            refuse_msg = self.system_messages.get("refuse_unsafe_query", ["Xin lỗi, tôi không thể hỗ trợ câu hỏi này."])
            if isinstance(refuse_msg, list):
                refuse_msg = random.choice(refuse_msg)
                
            result.update({
                "action": "reject_or_refuse",
                "status": action,
                "reasoning": f"Blocked by Hard Gate Rule: {reason}",
                "response": refuse_msg
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
            fallback_cf = self.settings.guardian_config.get('intent_fallback', {}).get('fallback_thresholds', {}).get('safe_confidence', 0.65)
            result["intent"] = {
                **intent_data,
                "intent": "safe",
                "confidence": max(confidence, fallback_cf),
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
            response_msg = self.system_messages.get(response_key, ["Tôi không thể trả lời câu hỏi này."])
            if isinstance(response_msg, list):
                response_msg = random.choice(response_msg)
            result["response"] = response_msg
            # Cập nhật lại status của result để báo hiệu hệ thống chặn
            result["status"] = intent_category
            
        result["reasoning"] = f"LLM Intent Classifier: {intent_data.get('reasoning', '')}"

        return result
