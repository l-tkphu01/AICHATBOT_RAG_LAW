import os
import sys
import time
import json
import re
import logging
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load môi trường
load_dotenv()

from app.utils.config_loader import Settings
from app.guardian.pipeline import GuardianPipeline
from app.retrieval.query_processor import QueryProcessor
from app.ingestion.embedder import EmbeddingGenerator
from app.ingestion.indexer import LegalIndexer
from app.retrieval.reranker import DocumentReranker
from app.generation.llm_client import chat_completion
from app.generation.generate import generate_answer

logger = logging.getLogger(__name__)

class LegalRAGPipeline:
    """
    RAG Pipeline Core Engine.
    Đóng gói toàn bộ 5 bước của quy trình:
    1. Guardian (Bảo mật & Phân tích Ý định)
    2. Query Processor (Viết lại & Mở rộng truy vấn)
    3. Retrieval (ChromaDB + Embedding)
    4. Reranker (Cohere Cross-Encoder)
    5. Generation (LLM Gemini)
    """
    _instance: Optional["LegalRAGPipeline"] = None

    @classmethod
    def get_instance(cls) -> "LegalRAGPipeline":
        """Get or create the singleton instance of the pipeline."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the pipeline components based on settings."""
        logger.info("Khởi tạo LegalRAGPipeline...")
        self.settings = Settings()
        
        self.guardian = GuardianPipeline()
        self.query_processor = QueryProcessor()
        self.embedder = EmbeddingGenerator()
        self.indexer = LegalIndexer()
        
        try:
            self.reranker = DocumentReranker()
            self.use_reranker = True
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Reranker: {e}", exc_info=True)
            self.use_reranker = False
            self.reranker = None
            
        self.fusion_top_k: int = getattr(self.settings, "fusion_top_k", 20)
        self.relevance_threshold: float = getattr(self.settings, "relevance_threshold", 0.05)
        logger.info("Hệ thống RAG sẵn sàng.")

    def _pick_prompt_text(self, key: str, default: str = "") -> str:
        """Resolve a prompt key that can be either a string or a list of candidate texts."""
        value = self.settings.get_prompt(key, default=default)
        if isinstance(value, list):
            return random.choice(value) if value else default
        if isinstance(value, str):
            return value
        return default

    def _soft_refusal_markers(self) -> Dict[str, List[str]]:
        prompts_cfg = getattr(self.settings, "prompts_config", {}).get("prompts", {})
        markers = prompts_cfg.get("soft_refusal_reason_markers", {})
        if isinstance(markers, dict):
            return {
                "greeting": [str(x).lower() for x in markers.get("greeting", []) if isinstance(x, str)],
                "small_talk": [str(x).lower() for x in markers.get("small_talk", []) if isinstance(x, str)],
            }
        return {"greeting": [], "small_talk": []}

    def _soft_refusal_fallback_text(self, reason: str, explicit_fallback: Optional[str] = None) -> str:
        if explicit_fallback and explicit_fallback.strip():
            return explicit_fallback.strip()

        prompts_cfg = getattr(self.settings, "prompts_config", {}).get("prompts", {})
        fallback_cfg = prompts_cfg.get("soft_refusal_fallbacks", {}) if isinstance(prompts_cfg, dict) else {}
        if isinstance(fallback_cfg, dict):
            value = fallback_cfg.get(reason)
            if isinstance(value, list) and value:
                return random.choice(value)
            if isinstance(value, str) and value.strip():
                return value.strip()

        reason_to_prompt_key = {
            "greeting": "soft_refusal_fallback_greeting",
            "small_talk": "soft_refusal_fallback_small_talk",
            "ambiguous_non_legal": "intent_clarify_message",
            "off_topic": "refuse_legal_scope",
            "unsafe_query": "refuse_unsafe_query",
        }
        prompt_key = reason_to_prompt_key.get(reason)
        if prompt_key:
            resolved = self._pick_prompt_text(prompt_key, "")
            if resolved.strip():
                return resolved.strip()

        return "Mình cần thêm thông tin để hỗ trợ đúng phạm vi pháp luật Việt Nam."

    def _normalize_soft_refusal_reason(self, query: str, status: str, intent_label: str) -> str:
        """Map runtime signals to stable soft-refusal reason labels used by prompt_soft_refusal."""
        q = (query or "").strip().lower()

        if status in {"unsafe", "blocked"}:
            return "unsafe_query"

        markers_cfg = self._soft_refusal_markers()
        greeting_markers = markers_cfg.get("greeting") or [
            "xin chào", "xin chao", "chào", "chao", "hello", "hi", "hey", "alo"
        ]
        small_talk_markers = markers_cfg.get("small_talk") or [
            "bạn khỏe", "ban khoe", "how are you", "cảm ơn", "cam on", "ok", "oke", "good morning", "good evening"
        ]

        if any(marker in q for marker in greeting_markers):
            return "greeting"
        if any(marker in q for marker in small_talk_markers):
            return "small_talk"

        if status in {"clarify", "uncertain", "unclear_intent"} or intent_label in {"clarify", "uncertain", "unclear_intent"}:
            return "ambiguous_non_legal"

        return "off_topic"

    @staticmethod
    def _normalize_guard_label(label: Any) -> str:
        """Normalize guardian labels to avoid branch misses caused by spaces/casing/punctuation."""
        if label is None:
            return ""
        text = str(label).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "outofscope": "out_of_scope",
            "out_of_scope.": "out_of_scope",
            "unclear": "unclear_intent",
            "ambiguous": "uncertain",
            "blocked": "unsafe",
        }
        return aliases.get(text, text)

    def _generate_soft_refusal(self, query: str, reason: str = "off_topic", fallback_answer: Optional[str] = None) -> str:
        """Sử dụng LLM nhỏ để sinh câu từ chối khéo léo (Soft Refusal)."""
        default_answer = self._soft_refusal_fallback_text(reason, explicit_fallback=fallback_answer)

        try:
            logger.info(f"Generating Soft Refusal for: '{reason}'")
            model_cfg = self.settings.resolve_ref("models_config.runtime.soft_refusal", default={})
            
            p_cfg = getattr(self.settings, 'prompts_config', {}).get('prompts', {})
                
            if reason == "unsafe_query":
                prompt_template = p_cfg.get("prompt_soft_refusal_unsafe", "")
                if prompt_template:
                    prompt = prompt_template.replace("{query}", query)
                else:
                    prompt = (
                        "<system>\n"
                        "Bạn là trợ lý AI tư vấn pháp luật Việt Nam. Nội dung người dùng vừa gửi đã bị hệ thống gắn nhãn unsafe.\n"
                        "Hãy từ chối dứt khoát nhưng lịch sự, không cung cấp bất kỳ hướng dẫn nào có thể gây hại hoặc vi phạm pháp luật.\n"
                        "Độ dài 1-2 câu, ngắn gọn, tự nhiên, không mở đầu bằng 'Xin lỗi' hoặc 'Rất tiếc'.\n"
                        "Kết thúc bằng lời mời đặt câu hỏi pháp lý hợp lệ.\n"
                        "</system>\n\n"
                        "<user_input>\n"
                        f"{query}\n"
                        "</user_input>"
                    )
            else:
                prompt_template = p_cfg.get("prompt_soft_refusal", "")
                if not prompt_template:
                    return default_answer
                prompt = prompt_template.replace("{query}", query).replace("{reason}", reason)
            
            provider = model_cfg.get("provider", "groq")
            model_id = model_cfg.get("model_id", "llama-3.1-8b-instant")
            
            resp = chat_completion(
                provider=provider,
                model_id=model_id,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý pháp luật Việt Nam, hãy tuân thủ đúng hướng dẫn system dưới đây."},
                    {"role": "user", "content": prompt}
                ],
                temperature=model_cfg.get("temperature", 0.6),
                max_tokens=model_cfg.get("max_tokens", 220)
            )

            # Current llm_client trả về chuẩn: {"text", "raw", ...}
            content = str(resp.get("text", "") or "").strip()

            # Backward-compatible fallback nếu adapter cũ trả payload kiểu choices/message/content
            if not content:
                content = str(
                    resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                ).strip()

            raw_payload = resp.get("raw", {}) if isinstance(resp, dict) else {}
            choices = raw_payload.get("choices", []) if isinstance(raw_payload, dict) else []
            finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
            if finish_reason == "length":
                logger.warning(
                    "Soft refusal completion hit token limit (finish_reason=length, max_tokens=%s)",
                    model_cfg.get("max_tokens", 220),
                )

            sanitized = self._sanitize_soft_refusal_text(content)
            return sanitized if sanitized else default_answer
        except Exception as e:
            logger.error(f"Soft refusal failed: {e}. Falling back to default message.")
            return default_answer

    @staticmethod
    def _sanitize_soft_refusal_text(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        cleaned = re.sub(
            r"</?(system|user_input|assistant|rules|output_schema|context)>",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^\s*system\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the end-to-step RAG pipeline for a given query.
        
        Args:
            query (str): The user query.
            
        Returns:
            Dict[str, Any]: The pipeline execution result including answer and sources.
        """
        result: Dict[str, Any] = {
            "query": query,
            "answer": "",
            "sources": [],
            "status": "success",
            "processing_time": 0.0,
            "metadata": {}
        }
        
        t0 = time.time()
        
        try:
            # 1. GUARDIAN
            guard_res = self.guardian.process_query(query)
            raw_status = guard_res.get("status", "unknown")
            raw_intent = guard_res.get("intent", {}).get("intent", "unknown")
            status = self._normalize_guard_label(raw_status)
            intent_label = self._normalize_guard_label(raw_intent)
            action = self._normalize_guard_label(guard_res.get("action", ""))
            reason_code = self._normalize_soft_refusal_reason(query, status, intent_label)
            
            result["metadata"]["guardian"] = {
                "status": status,
                "intent": intent_label,
                "action": action,
                "soft_reason": reason_code,
            }

            effective_status = status if status != "safe" else intent_label

            # TÁCH RIÊNG GREETING/SMALL TALK KHỎI OUT_OF_SCOPE
            if effective_status == "out_of_scope":
                result["status"] = "out_of_scope"
                # Luôn ưu tiên LLM để sinh câu trả lời tự nhiên; static response chỉ là fallback khi LLM lỗi.
                result["answer"] = self._generate_soft_refusal(
                    query,
                    reason_code,
                    fallback_answer=guard_res.get("response") or self._pick_prompt_text("refuse_legal_scope", ""),
                )
                result["processing_time"] = round(time.time() - t0, 3)
                return result

            # Các trường hợp đèn vàng: luôn ưu tiên LLM, static response chỉ dùng làm fallback.
            if effective_status in ["unclear_intent", "chitchat", "clarify", "uncertain"]:
                result["status"] = effective_status
                result["answer"] = self._generate_soft_refusal(
                    query,
                    reason_code,
                    fallback_answer=guard_res.get("response") or self._pick_prompt_text("intent_clarify_message", ""),
                )
                    
                result["processing_time"] = round(time.time() - t0, 3)
                return result

            # Xử lý ĐÈN ĐỎ (Hard Refusal): Bắt buộc chặn (mã độc, chửi thề, vi phạm cực đoan...)
            if status == "unsafe" or (action in ["reject_or_refuse", "blocked"] and effective_status != "out_of_scope"):
                result["status"] = "blocked"
                result["answer"] = self._generate_soft_refusal(
                    query,
                    "unsafe_query",
                    fallback_answer=guard_res.get("response")
                    or self._pick_prompt_text("refuse_unsafe_query", "Câu hỏi vi phạm chính sách an toàn."),
                )
                result["processing_time"] = round(time.time() - t0, 3)
                return result

            # 1.5 SUMMARIZE LONG QUERY (IF FLAGGED)
            if guard_res.get("action") == "summarize_then_classify":
                try:
                    logger.info(f"Summarizing long query ({len(query)} chars) via LLM...")
                    # 1. Láº¥y config tÃ³m táº¯t vÃ  prompt template
                    model_cfg = self.settings.resolve_ref("models_config.runtime.input_summarizer", default={})
                    
                    p_cfg = getattr(self.settings, 'prompts_config', {}).get('prompts', {})
                        
                    prompt_template = p_cfg.get("guard_input_summarizer", "")
                    if prompt_template:
                        prompt = prompt_template.replace("{query}", query)
                        
                        provider = model_cfg.get("provider", "groq")
                        model_id = model_cfg.get("model_id", "llama-3.1-8b-instant")
                        is_json = (model_cfg.get("response_format") == "json_object")
                        
                        s_resp = chat_completion(
                            provider=provider,
                            model_id=model_id,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=model_cfg.get("temperature", 0.1),
                            max_tokens=model_cfg.get("max_tokens", 256),
                            response_format={"type": "json_object"} if is_json else None
                        )
                        s_content = s_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Fallback nếu api trả về chuỗi json bị kẹp trong Markdown (vd: ```json {...} ```)
                        if s_content.startswith("```json"):
                            s_content = s_content[7:]
                            if s_content.endswith("```"):
                                s_content = s_content[:-3]
                        s_content = s_content.strip()
                        
                        # Nếu API trả về chuỗi trống hoặc lỗi, log ra và chuyển sang cắt xén thủ công
                        if not s_content:
                            logger.error(f"Summarizer API returned empty content. Fallback to truncation.")
                            query = query[:400] + " ... " + query[-400:]
                        else:
                            try:
                                s_json = json.loads(s_content)
                                if not s_json.get("in_scope", True):
                                    result["status"] = "out_of_scope"
                                    # Sinh câu từ chối khéo nếu summarize phát hiện ra là lạc đề từ đoạn dài
                                    result["answer"] = self._generate_soft_refusal(query, "off_topic")
                                    result["processing_time"] = round(time.time() - t0, 3)
                                    return result
                                    
                                summary_txt = s_json.get("summary", "")
                                    
                                if summary_txt:
                                    logger.info(f"Summarized query generated: '{summary_txt}'")
                                    result["metadata"]["guardian"]["summarized_query"] = summary_txt
                                    query = summary_txt
                                else:
                                    # Json hợp lệ nhưng field summary trống -> Cắt thủ công
                                    query = query[:400] + " ... " + query[-400:]
                                    
                            except json.JSONDecodeError as j_err:
                                logger.error(f"Summarizer generated invalid JSON: {s_content}. Fallback to truncation. Error: {j_err}")
                                query = query[:400] + " ... " + query[-400:]
                                
                except Exception as e:
                    logger.error(f"Failed to summarize query: {e}. Fallback to truncation.", exc_info=True)
                    # Chống sập diện rộng: Rót 2000 chữ vào RAG là thảm hoạ, nên ép cắt còn ~800 chữ dẫu API Llama 8B có sập
                    if len(query) > 1000:
                        query = query[:400] + " ... " + query[-400:]

            # 2. QUERY PROCESSOR
            expanded_queries = self.query_processor.generate_variants(query) or [query]
            result["metadata"]["expanded_queries"] = expanded_queries

            # 3. RETRIEVAL
            unique_docs: Dict[str, Any] = {}
            for eq in expanded_queries:
                try:
                    q_vec = self.embedder.embed_query(eq)
                    chunks = self.indexer.query_with_parent_context(
                        query_vector=q_vec, 
                        n_results=self.fusion_top_k, 
                        min_score=self.relevance_threshold
                    )
                    for c in chunks:
                        cid = c.get("id")
                        if cid and cid not in unique_docs:
                            unique_docs[cid] = c
                except Exception as e:
                    logger.error(f"Lỗi retrieval cho query '{eq}': {e}", exc_info=True)
            
            raw_chunks = list(unique_docs.values())
            
            if not raw_chunks:
                result["answer"] = self._pick_prompt_text(
                    "retrieval_no_context_message",
                    "Tôi chưa tìm thấy quy định pháp luật cụ thể nào trong cơ sở dữ liệu để trả lời câu hỏi này.",
                )
                result["processing_time"] = round(time.time() - t0, 3)
                return result

            # Khớp cấu trúc
            formatted_chunks: List[Dict[str, Any]] = []
            for c in raw_chunks:
                parent = c.get("parent") or {}
                doc_text = parent.get("text") or c.get("text") or ""
                meta = (c.get("metadata") or {}).copy()
                meta["chunk_id"] = c.get("id", "Unknown")
                meta["source"] = meta.get("law_name") or meta.get("document_id", "Không xác định")
                formatted_chunks.append({
                    "page_content": doc_text,
                    "text": doc_text,
                    "metadata": meta,
                    "score": c.get("score")
                })

            # 4. RERANKER
            reranked_chunks = formatted_chunks[:self.fusion_top_k]
            if self.use_reranker and self.reranker and formatted_chunks:
                try:
                    reranked_result = self.reranker.rerank(query, formatted_chunks)
                    reranked_chunks = reranked_result[:self.fusion_top_k]
                    if not reranked_chunks:
                        logger.info("Reranker trả về 0 chunks hợp lệ. Chuyển cho LLM xử lý Fallback.")
                except Exception as e:
                    logger.error(f"Lỗi reranker: {e}", exc_info=True)

            # Lưu lại sources vào kết quả
            for chunk in reranked_chunks[:5]:
                meta = chunk.get("metadata", {})
                score = chunk.get("rerank_score", chunk.get("score", 0.0))
                score_val = float(score) if score is not None else 0.0
                preview = (chunk.get("text") or chunk.get("page_content") or "").replace("\n", " ")[:200]
                result["sources"].append({
                    "chunk_id": meta.get("chunk_id", "unknown"),
                    "source_name": meta.get("source", "N/A"),
                    "preview_text": preview,
                    "score": round(score_val, 4)
                })

            # 5. GENERATION
            try:
                final_answer = generate_answer(query, reranked_chunks)
                result["answer"] = final_answer
            except Exception as e:
                logger.error(f"Lỗi Generation: {e}", exc_info=True)
                result["answer"] = self._pick_prompt_text(
                    "generation_error_message",
                    "Hệ thống sinh câu trả lời đang gặp sự cố. Vui lòng thử lại.",
                )
                result["status"] = "error"

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong Pipeline: {e}", exc_info=True)
            result["status"] = "error"
            result["answer"] = self._pick_prompt_text(
                "pipeline_unknown_error_message",
                "Đường truyền AI xảy ra lỗi không xác định.",
            )

        result["processing_time"] = round(time.time() - t0, 3)
        return result
