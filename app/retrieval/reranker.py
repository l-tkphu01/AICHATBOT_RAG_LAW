# app/retrieval/reranker.py
import os
import sys
import logging
import httpx
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.utils.config_loader import Settings

logger = logging.getLogger(__name__)

class DocumentReranker:
    """
    Uses Cohere Rerank API (or other provider) to accurately score and filter 
    retrieved chunks before passing to generation.
    """
    def __init__(self) -> None:
        self.settings = Settings()
        self.config: Dict[str, Any] = self.settings.reranker
        
        self.provider: str = self.config.get("provider", "cohere")
        self.model_name: str = self.config.get("model", "rerank-multilingual-v3.0")
        self.rerank_top_k: int = self.config.get("rerank_top_k", 5)
        self.score_threshold: float = float(self.config.get("score_threshold", 0.3))
        
        self.api_key: Optional[str] = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            logger.warning("COHERE_API_KEY could not be found in environment variables.")
            
        logger.info(f"Initialized Reranker client using API {self.provider.upper()} ({self.model_name})")

    def rerank(self, original_query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score a list of documents against a query using Cohere Rerank API.
        
        Args:
            original_query (str): The search query.
            documents (List[Dict[str, Any]]): List of document dictionaries containing text.
            
        Returns:
            List[Dict[str, Any]]: Top documents filtered by relevance score threshold.
        """
        if not documents:
            return []
            
        if not self.api_key:
            logger.warning("Missing API KEY -> Skipping Rerank step.")
            return documents[:self.rerank_top_k]

        docs_texts = [doc.get("text", "") for doc in documents]
        
        url = "https://api.cohere.com/v1/rerank"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "query": original_query,
            "documents": docs_texts,
            "top_n": len(documents),
            "return_documents": False 
        }

        logger.info(f"Reranking {len(documents)} snippets via Cohere...")
        
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                results_array = result.get("results", [])
                
                scored_docs: List[Dict[str, Any]] = []
                for item in results_array:
                    original_idx = int(item["index"])
                    relevance_score = float(item["relevance_score"])
                    
                    doc_item = documents[original_idx]
                    doc_item["rerank_score"] = relevance_score
                    scored_docs.append(doc_item)
                
                scored_docs.sort(key=lambda x: float(x.get("rerank_score", 0)), reverse=True)
                
                highest_score = scored_docs[0]["rerank_score"] if scored_docs else 0.0
                logger.info(f"Highest Rerank Score: {highest_score:.4f} (Threshold: {self.score_threshold})")
                
                filtered_docs = [d for d in scored_docs if d.get("rerank_score", 0) >= self.score_threshold]
                
                final_docs = filtered_docs[:self.rerank_top_k]
                deduped_docs: List[Dict[str, Any]] = []
                seen_keys = set()
                for doc in final_docs:
                    meta = doc.get("metadata") or {}
                    dedup_key = meta.get("chunk_id") or doc.get("id") or doc.get("text") or ""
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)
                    deduped_docs.append(doc)
                
                logger.info(f"Cohere selected {len(deduped_docs)} verified chunks (API Time: {response.elapsed.total_seconds():.2f}s)")
                return deduped_docs
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Cohere API HTTP Error {e.response.status_code}: {e.response.text}", exc_info=True)
        except Exception as e:
            logger.error(f"Error executing Cohere reranking: {str(e)}", exc_info=True)
            
        return documents[:self.rerank_top_k]
