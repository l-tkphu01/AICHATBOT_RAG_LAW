# app/retrieval/hybrid_retriever.py
# Module truy xuất kết hợp: Dense (Vector) + Sparse (BM25) + Metadata Filter + RRF Fusion
import os
import sys
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.utils.config_loader import Settings

class HybridRetriever:
    """
    Hybrid Retrieval system using Dense (Vector) and Sparse (BM25) search.
    Fuses results using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self, 
        chroma_collection: Optional[Any] = None, 
        bm25_corpus: Optional[Any] = None, 
        chunk_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the HybridRetriever.
        """
        self.settings = Settings()
        self.config = self.settings.config["shared_configs"]["query"]["retrieval"]
        
        self.dense_top_k: int = self.config.get("dense_top_k", 15)
        self.sparse_top_k: int = self.config.get("sparse_top_k", 15)
        self.fusion_top_k: int = self.config.get("fusion_top_k", 20)
        self.rrf_k: int = self.config.get("rrf_k", 60)
        
        self.collection = chroma_collection
        self.bm25 = bm25_corpus
        self.chunk_dict = chunk_dict or {}

    def _dense_search_multi(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Vector search via ChromaDB for multiple query variants.
        """
        if not self.collection:
            return []

        results = self.collection.query(
            query_texts=queries,
            n_results=self.dense_top_k
        )
        
        unique_docs: Dict[str, Dict[str, Any]] = {}
        for i in range(len(queries)):
            for j in range(len(results["ids"][i])):
                doc_id = results["ids"][i][j]
                distance = results["distances"][i][j]
                
                if doc_id not in unique_docs or distance < unique_docs[doc_id]['distance']:
                    unique_docs[doc_id] = {
                        "doc_id": doc_id,
                        "distance": distance,
                        "text": results["documents"][i][j],
                        "metadata": results["metadatas"][i][j]
                    }
                    
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x["distance"])
        return sorted_docs[:self.dense_top_k]

    def _sparse_search_multi(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        BM25 keyword search using query variants.
        """
        if not self.bm25 or not self.chunk_dict:
            return []

        combined_query = " ".join(queries).lower()
        tokenized_query = combined_query.split()
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        top_n_indexes = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:self.sparse_top_k]
        
        bm25_results: List[Dict[str, Any]] = []
        for idx in top_n_indexes:
            if doc_scores[idx] <= 0:
                continue
            
            doc_id = list(self.chunk_dict.keys())[idx]
            bm25_results.append({
                "doc_id": doc_id,
                "score": float(doc_scores[idx]),
                "text": self.chunk_dict[doc_id]["text"],
                "metadata": self.chunk_dict[doc_id]["metadata"]
            })
            
        return bm25_results

    def reciprocal_rank_fusion(
        self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine Dense and Sparse results using Reciprocal Rank Fusion (RRF).
        """
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        for rank, doc in enumerate(dense_results):
            doc_id = doc["doc_id"]
            doc_map[doc_id] = doc
            fused_scores[doc_id] = 1.0 / (self.rrf_k + rank + 1)
            
        for rank, doc in enumerate(sparse_results):
            doc_id = doc["doc_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                
            fused_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            
        sorted_fused_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_results: List[Dict[str, Any]] = []
        for doc_id, rrf_score in sorted_fused_docs[:self.fusion_top_k]:
            doc_data = doc_map[doc_id]
            doc_data["rrf_score"] = rrf_score
            final_results.append(doc_data)
            
        return final_results

    def retrieve(self, multi_queries: List[str]) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval pipeline.
        Returns top chunks fused via RRF.
        """
        print(f"[-] Đang lấy '{self.dense_top_k}' chunks từ Vector (ChromaDB)...")
        dense_hits = self._dense_search_multi(multi_queries)
        
        print(f"[-] Đang lấy '{self.sparse_top_k}' chunks từ Keyword (BM25)...")
        sparse_hits = self._sparse_search_multi(multi_queries)
        
        print(f"[-] Tiến hành trộn RRF (k={self.rrf_k}) để chắt lọc '{self.fusion_top_k}' chunks tinh túy...")
        final_docs = self.reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        return final_docs
