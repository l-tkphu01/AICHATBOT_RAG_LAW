import json
import re
import logging
from typing import List, Dict, Any, Optional
from app.utils.config_loader import Settings
from app.generation.llm_client import chat_completion

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Module that rewriting and expanding the user's query 
    to retrieve better context components.
    """
    def __init__(self) -> None:
        self.settings = Settings()
        
        query_processing_cfg = getattr(self.settings, 'query_processing', self.settings.config.get('query_processing', {}))
        self.flow_cfg: Dict[str, Any] = query_processing_cfg.get('flow', {})
        self.rewrite_cfg: Dict[str, Any] = query_processing_cfg.get('rewrite', {})
        
        self.enabled: bool = self.rewrite_cfg.get('enabled', True)
        self.max_variants: int = self.rewrite_cfg.get('max_variants', 3)

        model_ref = self.rewrite_cfg.get('multi_query_model_ref') or self.rewrite_cfg.get('model_ref')
        self.llm_cfg: Dict[str, Any] = self.settings.resolve_ref(
            model_ref,
            default=self.settings.resolve_ref('models_config.runtime.query_rewriter', default={}),
        )
        if not isinstance(self.llm_cfg, dict):
            self.llm_cfg = {}

        self.provider: str = self.llm_cfg.get('provider', 'groq')
        self.model_name: str = self.llm_cfg.get('model_id', 'llama-3.1-8b-instant')  
        self.temperature: float = float(self.llm_cfg.get('temperature', 0.2))

        self.fallback_provider: Optional[str] = self.llm_cfg.get('fallback_provider', 'openrouter')
        self.fallback_model: Optional[str] = self.llm_cfg.get('fallback_model', 'google/gemini-2.5-flash-lite')

        prompt_key = self.rewrite_cfg.get('prompt_keys', {}).get('multi')
        if not isinstance(prompt_key, str) or not prompt_key:
            rewrite_stage = self.flow_cfg.get('rewrite_stage', 'prompts_config.legal_query_rewrite_multi')
            if isinstance(rewrite_stage, str) and rewrite_stage.startswith('prompts_config'):
                raw_prompt = self.settings.resolve_ref(
                    rewrite_stage,
                    default='Bạn là một thư ký pháp lý AI. PHẢI TRẢ VỀ JSON: {"variants": ["câu 1"]}',
                )
            else:
                raw_prompt = self.settings.get_prompt(
                    'legal_query_rewrite_multi',
                    'Bạn là một thư ký pháp lý AI. PHẢI TRẢ VỀ JSON: {"variants": ["câu 1"]}',
                )
        else:
            raw_prompt = self.settings.get_prompt(
                prompt_key,
                'Bạn là một thư ký pháp lý AI. PHẢI TRẢ VỀ JSON: {"variants": ["câu 1"]}',
            )

        if not isinstance(raw_prompt, str):
            raw_prompt = 'Bạn là một thư ký pháp lý AI. PHẢI TRẢ VỀ JSON: {"variants": ["câu 1"]}'

        self.system_prompt: str = raw_prompt.replace('{max_variants}', str(self.max_variants))

    def generate_variants(self, original_query: str) -> List[str]:
        """
        Generate query variants using an LLM configured for rewrite tasks.
        
        Args:
            original_query (str): The raw input query.
            
        Returns:
            List[str]: Original query and its rewritten variants.
        """
        if not self.enabled:
            return [original_query]

        exact_article_hint = self._build_exact_article_hint(original_query)

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': f'Câu hỏi gốc: {original_query}'}   
        ]

        def _parse_variants(content: str) -> List[str]:
            content = content.replace('```json', '').replace('```', '').strip() 
            try:
                data = json.loads(content)
                return data.get('variants', data.get('queries', []))
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from query rewriting response.")
                return []

        logger.info(f"QueryProcessor is expanding query: '{original_query}'")

        variants: List[str] = []
        try:
            response = chat_completion(
                provider=self.provider,
                model_id=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format={'type': 'json_object'} if self.provider == 'groq' else None
            )
            variants = _parse_variants(response.get('text', '{}'))

        except Exception as e:
            logger.warning(f"Error with primary provider ({self.provider}): {e}")

            if self.fallback_provider and self.fallback_model:
                try:
                    logger.info(f"Using fallback provider ({self.fallback_provider}) for query rewriting.")
                    fallback_response = chat_completion(
                        provider=self.fallback_provider,
                        model_id=self.fallback_model,
                        messages=messages,
                        temperature=self.temperature
                    )
                    variants = _parse_variants(fallback_response.get('text', '{}'))
                except Exception as fb_err:
                    logger.error(f"Fallback provider failed: {fb_err}", exc_info=True)

        final_queries = [original_query]
        if exact_article_hint and exact_article_hint.lower() != original_query.lower():
            final_queries.append(exact_article_hint)
        if isinstance(variants, list) and len(variants) > 0:
            for q in variants:
                if isinstance(q, str) and q.strip() and q.lower() != original_query.lower():
                    final_queries.append(q.strip())

        return final_queries[:self.max_variants + 1]

    def _build_exact_article_hint(self, original_query: str) -> str:
        """
        Extract exact article numbers to construct a precise query hint.
        """
        match = re.search(r"(?:điều|điều\s+)[\s]*([0-9]{1,3})", original_query, flags=re.IGNORECASE)
        if not match:
            return ""

        article_number = match.group(1)
        base_query = original_query.strip().rstrip("?.! ")
        if f"Điều {article_number}" not in base_query and f"điều {article_number}" not in base_query.lower():
            return f"Điều {article_number} Luật Quản lý thuế 2019 nội dung quy định và căn cứ pháp lý"

        return f"Điều {article_number} Luật Quản lý thuế 2019 nội dung quy định và căn cứ pháp lý"
