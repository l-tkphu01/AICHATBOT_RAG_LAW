import os
import sys
import time
import logging
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
            status = guard_res.get("status", "unknown")
            intent_label = guard_res.get("intent", {}).get("intent", "unknown")
            
            result["metadata"]["guardian"] = {
                "status": status,
                "intent": intent_label
            }

            if status != "safe" or guard_res.get("action") == "reject_or_refuse":
                result["status"] = "blocked"
                result["answer"] = guard_res.get("response") or "Câu hỏi vi phạm chính sách an toàn."
                result["processing_time"] = round(time.time() - t0, 3)
                return result

            if intent_label == "out_of_scope":
                result["status"] = "out_of_scope"
                result["answer"] = guard_res.get("response") or "Câu hỏi nằm ngoài phạm vi tư vấn pháp luật."
                result["processing_time"] = round(time.time() - t0, 3)
                return result

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
                result["answer"] = ("Tôi chưa tìm thấy quy định pháp luật cụ thể nào "
                                   "trong cơ sở dữ liệu để trả lời câu hỏi này.")
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
                result["answer"] = "Xin lỗi, hệ thống sinh câu trả lời đang gặp sự cố. Vui lòng thử lại."
                result["status"] = "error"

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong Pipeline: {e}", exc_info=True)
            result["status"] = "error"
            result["answer"] = "Đường truyền AI xảy ra lỗi không xác định."

        result["processing_time"] = round(time.time() - t0, 3)
        return result
