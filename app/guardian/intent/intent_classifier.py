import json
from typing import Dict, Any

from app.generation.llm_client import chat_completion
from app.utils.config_loader import Settings


class IntentClassifier:
    def __init__(self):
        self.settings = Settings()
        flow_cfg = self.settings.query_processing.get('flow', {}) if isinstance(self.settings.query_processing, dict) else {}
        intent_model_ref = flow_cfg.get('intent_stage', 'models_config.runtime.guard')

        intent_cfg = self.settings.resolve_ref(
            intent_model_ref,
            default=self.settings.resolve_ref('models_config.runtime.guard', default={}),
        )

        if isinstance(intent_cfg, dict):
            self.model_name = intent_cfg.get('model_id', 'llama-3.3-70b-versatile')
            self.provider = intent_cfg.get('provider', 'groq')
            self.temperature = intent_cfg.get('temperature', 0.1)
            # Fallback model nếu Groq gọi Llama tạch do Rate Limit / Hết timeout
            self.fallback_provider = intent_cfg.get('fallback_provider', 'groq')
            self.fallback_model = intent_cfg.get('fallback_model', 'qwen-2.5-32b') # Tên model chuẩn của qwen trên groq thường là qwen-2.5-32b
        else:
            self.model_name = 'llama-3.3-70b-versatile'
            self.provider = 'groq'
            self.temperature = 0.1
            self.fallback_provider = 'groq'
            self.fallback_model = 'qwen-2.5-32b'

        intent_fallback_cfg = self.settings.guardian_config.get('intent_fallback', {}) if isinstance(self.settings.guardian_config, dict) else {}
        prompt_key = intent_fallback_cfg.get('prompt_key', 'intent_fallback_classify_vn')
        self.system_prompt = self.settings.get_prompt(
            str(prompt_key),
            'You are an Intent Classifier. Ensure to output JSON: {"intent": "good_intent", "confidence": 1.0, "reasoning": "..."}',
        )
        
        fallback_keywords = self.settings.guardian_config.get('intent_fallback', {}).get('fallback_keywords', {})
        self.legal_keywords = fallback_keywords.get('legal_domain', [])
        self.out_of_scope_keywords = fallback_keywords.get('out_of_scope', [])

    def _looks_like_legal_query(self, query: str) -> bool:
        # Bắt keyword đôi khi làm bypass LLM đối với những câu ngắn/mơ hồ
        # Nên chỉ dùng heuristict này nếu query đủ dài hoặc chứa nhiều keyword pháp lý rõ ràng
        normalized = query.lower()
        words = normalized.split()
        
        # Nếu câu quá ngắn (< 10 từ) và chỉ có 1 keyword, nên để LLM đánh giá intent
        matched_keywords = [kw for kw in self.legal_keywords if kw in normalized]
        if len(words) < 10 or len(matched_keywords) <= 1:
            return False
            
        return len(matched_keywords) > 0
    def _looks_out_of_scope(self, query: str) -> bool:
        normalized = query.lower()
        return any(keyword in normalized for keyword in self.out_of_scope_keywords)

    def _normalize_llm_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        intent = payload.get('intent') or payload.get('intent_label')
        safety = payload.get('safety_label')
        confidence = payload.get('confidence', 0.0)

        if isinstance(safety, str):
            safety = safety.lower()
        if isinstance(intent, str):
            intent = intent.lower()

        if safety == 'unsafe' or intent == 'bad_intent':
            return {
                'intent': 'unsafe',
                'confidence': confidence,
                'reasoning': payload.get('reasoning') or payload.get('reason') or 'model flagged unsafe',
            }

        if safety == 'out_of_scope':
            return {
                'intent': 'out_of_scope',
                'confidence': confidence,
                'reasoning': payload.get('reasoning') or payload.get('reason') or 'model flagged out of scope',
            }

        if intent in {'clarify', 'needs_more_context', 'unknown', 'uncertain'}:
            return {
                'intent': 'clarify',
                'confidence': confidence,
                'reasoning': payload.get('reasoning') or payload.get('reason') or 'model requested clarification or unknown intent',
            }

        if intent in {'safe', 'good_intent'}:
            return {
                'intent': 'safe',
                'confidence': confidence,
                'reasoning': payload.get('reasoning') or payload.get('reason') or 'model flagged safe',
            }

        return {
            'intent': 'uncertain',
            'confidence': confidence,
            'reasoning': payload.get('reasoning') or payload.get('reason') or 'unrecognized model output',
        }

    def classify(self, query: str, features: Dict[str, float] = None) -> Dict[str, Any]:
        if self._looks_out_of_scope(query):
            return {
                'intent': 'out_of_scope',
                'confidence': 0.9,
                'reasoning': 'Rule-based out-of-scope match',
                'safety_flag': False,
                'suggested_action': 'reject_or_refuse',
            }

        if self._looks_like_legal_query(query):
            return {
                'intent': 'safe',
                'confidence': 0.85,
                'reasoning': 'Rule-based legal-domain match',
                'safety_flag': True,
                'suggested_action': 'continue_pipeline',
            }

        prompt = f'User Query: {query}'

        def _call_model(provider: str, model_id: str) -> Dict[str, Any]:
            response = chat_completion(
                provider=provider,
                model_id=model_id,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=self.temperature,
                response_format={'type': 'json_object'} if provider == 'groq' else None
            )

            content = response.get('text', '{}')
            content = content.replace('```json', '').replace('```', '').strip()

            result_json = json.loads(content)
            if not isinstance(result_json, dict):
                return {
                    'intent': 'uncertain',
                    'confidence': 0.0,
                    'reasoning': f'LLM output not a dict ({provider})',
                    'safety_flag': False,
                    'suggested_action': 'request_more_context',
                }
            return self._normalize_llm_result(result_json)

        try:
            # Lần 1: Gọi Provider chính
            return _call_model(self.provider, self.model_name)
            
        except Exception as e_primary:
            # Lần 2: NẾU THẤT BẠI, GỌI FALLBACK PROVIDER TỪ YAML
            try:
                result_fallback = _call_model(self.fallback_provider, self.fallback_model)
                result_fallback['reasoning'] = f"[Used {self.fallback_provider} Fallback] " + result_fallback.get('reasoning', '')
                return result_fallback
            except Exception as e_fallback:
                # Nếu cà 2 đường truyền cùng ngỏm thì mới thua
                if self._looks_like_legal_query(query):
                    return {
                        'intent': 'safe',
                        'confidence': 0.6,
                        'reasoning': f'Both LLM providers failed, using legal heuristic: Primary={str(e_primary)}, Fallback={str(e_fallback)}',
                        'safety_flag': True,
                        'suggested_action': 'continue_pipeline',
                    }

                return {
                    'intent': 'uncertain',
                    'confidence': 0.0,
                    'reasoning': f'All LLM APIs failed. Primary={str(e_primary)}, Fallback={str(e_fallback)}',
                    'safety_flag': False,
                    'suggested_action': 'reject_or_refuse'
                }
