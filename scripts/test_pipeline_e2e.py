import sys
import os
import time
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config_loader import Settings
from app.guardian.pipeline import GuardianPipeline
from app.retrieval.query_processor import QueryProcessor
from app.ingestion.embedder import EmbeddingGenerator
from app.ingestion.indexer import LegalIndexer
from app.retrieval.reranker import DocumentReranker
from app.generation.llm_client import chat_completion

class RAGPipeline:
    def __init__(self):
        print("\n\033[1;36m" + "="*60 + "\033[0m")
        print("\033[1;32m[+] KHỞI TẠO HỆ THỐNG RAG PIPELINE THỬ NGHIỆM\033[0m")
        print("\033[1;36m" + "="*60 + "\033[0m")
        
        t0 = time.time()
        self.settings = Settings()
        
        print("  [1/5] Khởi tạo Guardian (Bảo mật & Ý định)...")
        self.guardian = GuardianPipeline()
        
        print("  [2/5] Khởi tạo Query Processor (Multi-query)...")
        self.query_processor = QueryProcessor()
        
        print("  [3/5] Khởi tạo Embedder (Mã hóa Vector)...")
        self.embedder = EmbeddingGenerator()
        
        print("  [4/5] Khởi tạo VectorStore (ChromaDB + Hydration)...")
        self.indexer = LegalIndexer()
        
        print("  [5/5] Khởi tạo Reranker (Cross-Encoder)...")
        try:
            self.reranker = DocumentReranker()
            self.use_reranker = True
        except Exception as e:
            print(f" [!] Cảnh báo Reranker: {e} -> Sẽ bypass bước này.")
            self.use_reranker = False
        
        self.fusion_top_k = self.settings.fusion_top_k
        print(f"\033[1;32m  [HOÀN TẤT] Hệ thống online sau {(time.time()-t0):.2f}s.\033[0m\n")

    @staticmethod
    def _normalize_guard_label(label: Any) -> str:
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

    @staticmethod
    def _normalize_soft_refusal_reason(query: str, status: str, intent_label: str) -> str:
        q = (query or "").strip().lower()

        if status in {"unsafe", "blocked"}:
            return "unsafe_query"

        greeting_markers = [
            "xin chào", "xin chao", "chào", "chao", "hello", "hi ", " hi", "hey", "alo"
        ]
        small_talk_markers = [
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
    def _soft_refusal_fallback(reason: str) -> str:
        fallback_map = {
            "greeting": "Xin chào! Mình là trợ lý tư vấn pháp luật Việt Nam, bạn đang cần hỗ trợ vấn đề pháp lý nào?",
            "small_talk": "Mình tập trung hỗ trợ pháp luật Việt Nam. Bạn có tình huống pháp lý nào cần mình tra cứu không?",
            "ambiguous_non_legal": "Bạn mô tả thêm bối cảnh để mình xác định đúng khía cạnh pháp lý và tra đúng điều luật nhé?",
            "off_topic": "Mảng này nằm ngoài phạm vi hỗ trợ của mình. Nếu bạn cần tư vấn pháp luật Việt Nam, mình sẵn sàng hỗ trợ.",
            "unsafe_query": "Nội dung này mình không thể hỗ trợ. Nếu bạn cần tư vấn pháp lý hợp lệ, mình sẵn sàng hỗ trợ.",
        }
        return fallback_map.get(reason, fallback_map["off_topic"])

    def _generate_soft_refusal(self, query: str, reason: str = "off_topic", fallback_answer: str = "") -> str:
        default_answer = (fallback_answer or "").strip() or self._soft_refusal_fallback(reason)
        try:
            model_cfg = getattr(self.settings, "models_config", {}).get("runtime", {}).get("soft_refusal", {})
            prompt_cfg = getattr(self.settings, "prompts_config", {}).get("prompts", {})

            if reason == "unsafe_query":
                prompt_template = prompt_cfg.get("prompt_soft_refusal_unsafe", "")
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
                prompt_template = prompt_cfg.get("prompt_soft_refusal", "")
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

            content = str(resp.get("text", "") or "").strip()
            if not content:
                content = str(resp.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

            sanitized = self._sanitize_soft_refusal_text(content)
            return sanitized or default_answer
        except Exception:
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

    def chat(self, query: str):
        print(f"\n\033[1;35m{'='*70}\033[0m")
        print(f"\033[1;37;44m  NGƯỜI DÙNG: \033[0m \033[1;33m{query}\033[0m")
        print(f"\033[1;35m{'='*70}\033[0m")

        t_total = time.time()

        # =========================================================
        # BƯỚC 1: BỘ LỌC GUARDIAN
        # =========================================================
        print("\n\033[1;34m[STEP 1] BỘ LỌC GUARDIAN (An toàn & Phân loại)\033[0m")
        tg = time.time()
        try:
            guard_res = self.guardian.process_query(query)
            
            # Xử lý dict siêu an toàn
            status = self._normalize_guard_label(guard_res.get("status", "unknown"))
            intent_info = guard_res.get("intent") or {}
            intent_label = self._normalize_guard_label(intent_info.get("intent", "unknown"))
            action = self._normalize_guard_label(guard_res.get("action", ""))
            confidence = intent_info.get("confidence", 0.0)
            reason = guard_res.get("reasoning", "")
            bot_msg = guard_res.get("response", "")
            effective_status = status if status != "safe" else intent_label
            reason_code = self._normalize_soft_refusal_reason(query, status, intent_label)

            print(f"  Tốn: {time.time() - tg:.2f}s")
            print(f"  Ý định: {intent_label.upper()} ({(confidence*100):.1f}%) | Trạng thái: {status.upper()} | Action: {action.upper() if action else 'N/A'}")

            if effective_status == "out_of_scope":
                display_msg = self._generate_soft_refusal(
                    query,
                    reason_code,
                    fallback_answer=bot_msg or reason or self._soft_refusal_fallback("off_topic"),
                )
                print(f"\n\033[1;43m  YÊU CẦU LÀM RÕ/TỪ CHỐI NHẸ \033[0m \033[1;33m{display_msg}\033[0m")
                return

            if effective_status in ["unclear_intent", "chitchat", "clarify", "uncertain"]:
                display_msg = self._generate_soft_refusal(
                    query,
                    reason_code,
                    fallback_answer=bot_msg or reason or self._soft_refusal_fallback("ambiguous_non_legal"),
                )
                print(f"\n\033[1;43m  YÊU CẦU LÀM RÕ/TỪ CHỐI NHẸ \033[0m \033[1;33m{display_msg}\033[0m")
                return

            if status == "unsafe" or action in ["reject_or_refuse", "blocked"]:
                display_msg = self._generate_soft_refusal(
                    query,
                    "unsafe_query",
                    fallback_answer=bot_msg or reason or self._soft_refusal_fallback("unsafe_query"),
                )
                print(f"\n\033[1;41m  BỊ CHẶN \033[0m \033[1;31m{display_msg}\033[0m")
                return
        except Exception as e:
            print(f"\033[1;31m[LỖI GUARDIAN]: {e}\033[0m")
            return

        # =========================================================
        # BƯỚC 2: MỞ RỘNG TRUY VẤN
        # =========================================================
        print("\n\033[1;34m[STEP 2] QUERY PROCESSOR (Viết lại & Tách ý)\033[0m")
        tq = time.time()
        try:
            expanded_queries = self.query_processor.generate_variants(query)
            print(f"  Tốn: {time.time() - tq:.2f}s")
            print(f"  Mở rộng thành {len(expanded_queries)} ngữ cảnh tìm kiếm:")
            for i, eq in enumerate(expanded_queries):
                print(f"    {i}. {eq}")
        except Exception as e:
            print(f"\033[1;31m[LỖI QUERY_EXPANSION]: {e}\033[0m")
            expanded_queries = [query]

        # =========================================================
        # BƯỚC 3: TRUY XUẤT DỮ LIỆU VECTOR
        # =========================================================
        print("\n\033[1;34m[STEP 3] RETRIEVAL (Quét CSDL & Hydrate Parent)\033[0m")
        tr = time.time()
        raw_docs_dict = {}
        try:
            for eq in expanded_queries:
                q_vec = self.embedder.embed_query(eq)
                results = self.indexer.query_with_parent_context(
                    query_vector=q_vec, 
                    n_results=self.settings.fusion_top_k, 
                    min_score=self.settings.relevance_threshold
                )
                # Chống trùng lặp đoạn văn
                for r in results:
                    uid = r.get("id")
                    if uid and uid not in raw_docs_dict:
                        raw_docs_dict[uid] = r
        except Exception as e:
            print(f"\033[1;31m[LỖI VECTOR_SEARCH]: {e}\033[0m")
            
        raw_chunks = list(raw_docs_dict.values())
        print(f"  Tốn: {time.time() - tr:.2f}s")
        print(f"  Quét trúng tổng {len(raw_chunks)} đoạn luật gốc hợp lệ.")

        if not raw_chunks:
            print("\n\033[1;43m  TRỐNG DATA \033[0m \033[1;33mHệ thống chuyển sang fallback.\033[0m")
            fallback_msg = guard_res.get("response") or "Tôi chưa tìm thấy quy định pháp luật cụ thể nào trong cơ sở dữ liệu để trả lời câu hỏi này."
            self._print_result(fallback_msg, time.time() - t_total)
            return

        # Đóng gói an toàn để xử lý triệt để Object NoneType
        formatted_chunks = []
        for c in raw_chunks:
            parent = c.get("parent") or {}
            doc_text = parent.get("text") or c.get("text") or ""
            
            meta = (c.get("metadata") or {}).copy()
            meta["chunk_id"] = c.get("id", "Unknown")
            meta["source"] = meta.get("law_name") or meta.get("document_id", "Không xác định nguồn")
            
            formatted_chunks.append({
                "page_content": doc_text,
                "text": doc_text,  # Dành cho Cohere Reranker cần key 'text'
                "metadata": meta
            })

        # =========================================================
        # BƯỚC 4: RERANK TÀI LIỆU
        # =========================================================
        print("\n\033[1;34m[STEP 4] RERANKER (Sàng lọc chéo Cross-Encoder)\033[0m")
        trr = time.time()
        reranked_chunks = formatted_chunks[:self.fusion_top_k]
        if self.use_reranker and len(formatted_chunks) > 0:
            try:
                reranked_result = self.reranker.rerank(query, formatted_chunks)
                reranked_chunks = reranked_result[:self.fusion_top_k]
                print(f"  Tốn: {time.time() - trr:.2f}s")
                print(f"  Chọn lọc ra TOP {len(reranked_chunks)} documents sát nghĩa nhất.")
                if not reranked_chunks:
                    print(" \033[1;33m[!] Reranker trả về 0 chunks hợp lệ -> Chuyển cho LLM xử lý Fallback (No-Context).\033[0m")
            except Exception as e:
                print(f" \033[1;31m[LỖI RERANK]: {e} \033[0m")
                reranked_chunks = formatted_chunks[:self.fusion_top_k]
        else:
            print(" [!] Bỏ qua reranker -> dùng trực tiếp chunks đã retrieve.")

        print("  \033[1;36m[TOP CHUNKS ĐƯỢC CHỌN]\033[0m")
        for i, chunk in enumerate(reranked_chunks[:5], 1):
            meta = chunk.get("metadata", {})
            source = meta.get("source") or meta.get("law_name") or meta.get("document_id") or "Unknown"
            score = chunk.get("rerank_score", chunk.get("score"))
            score_text = f"{float(score):.4f}" if score is not None else "n/a"
            preview = (chunk.get("text") or chunk.get("page_content") or "").replace("\n", " ")[:150]
            print(f"    {i}. Điểm={score_text} | Nguồn={source}\n       Trích đoạn: \"{preview}...\"\n")

        print(f"  └─ Context đưa vào generation: {len(reranked_chunks)} chunks.")

        # =========================================================
        # BƯỚC 5: TẠO VĂN BẢN TRẢ LỜI
        # =========================================================
        print("\n\033[1;34m[STEP 5] GENERATION (ĐÃ VÔ HIỆU HÓA)\033[0m")
        tgen = time.time()
        final_answer = (
            "LLM generation đã được vô hiệu hóa trong scripts/test_pipeline_e2e.py.\n"
            f"Số chunks sau rerank: {len(reranked_chunks)}.\n"
            "Xem mục [TOP CHUNKS ĐƯỢC CHỌN] để đánh giá chất lượng truy xuất."
        )
        print(f"  Tốn: {time.time() - tgen:.2f}s")

        self._print_result(final_answer, time.time() - t_total)

    def _print_result(self, answer: str, elapsed: float):
        print(f"\n\033[1;32m{'='*25} TRẢ LỜI CỦA AI {'='*25}\033[0m")
        print(f"\033[1;36m Tổng thời gian chạy pipeline: {elapsed:.2f} giây\033[0m")
        print("-" * 65)
        print(f"\n{answer}\n")
        print("=" * 66)

if __name__ == '__main__':
    pipeline = RAGPipeline()
    while True:
        try:
            q = input("\n\033[1;36m[BẠN] \033[0m ").strip()
            if not q: continue
            if q.lower() in ('exit', 'quit', 'q'):
                print("\033[1;33mTạm biệt!\033[0m")
                break
            pipeline.chat(q)
        except KeyboardInterrupt:
            print("\n\033[1;33mHủy truy vấn. Thoát.\033[0m")
            break
        except Exception as e:
            print(f"\n\033[1;31m[LỖI VÒNG LẶP]: {e}\033[0m")

