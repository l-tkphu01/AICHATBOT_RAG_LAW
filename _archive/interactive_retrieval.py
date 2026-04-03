import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config_loader import Settings
from app.guardian.pipeline import GuardianPipeline
from app.retrieval.query_processor import QueryProcessor
from app.ingestion.embedder import EmbeddingGenerator
from app.ingestion.indexer import LegalIndexer
from app.retrieval.reranker import DocumentReranker
from app.generation.generate import generate_answer


class InteractiveRetrievalTest:
    def __init__(self):
        load_dotenv()

        print("=" * 72)
        print("[+] INTERACTIVE RAG RETRIEVAL TEST")
        print("=" * 72)

        self.settings = Settings()

        print("[1/5] Khởi tạo Guardian...")
        self.guardian = GuardianPipeline()

        print("[2/5] Khởi tạo Query Processor...")
        self.query_processor = QueryProcessor()

        print("[3/5] Khởi tạo Embedder...")
        self.embedder = EmbeddingGenerator()

        print("[4/5] Khởi tạo Vector Index...")
        self.indexer = LegalIndexer()

        print("[5/5] Khởi tạo Reranker...")
        try:
            self.reranker = DocumentReranker()
            self.use_reranker = True
        except Exception as exc:
            print(f"[!] Reranker unavailable: {exc}")
            self.use_reranker = False
            self.reranker = None

        self.fusion_top_k = getattr(self.settings, "fusion_top_k", 20)
        self.relevance_threshold = getattr(self.settings, "relevance_threshold", 0.3)
        print("[✓] Hệ thống sẵn sàng\n")

    def _print_chunks(self, chunks):
        print(f"  └─ Có {len(chunks)} chunk được đưa vào generation:")
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata") or {}
            source = meta.get("source") or meta.get("law_name") or meta.get("document_id") or "Unknown"
            score = chunk.get("rerank_score", chunk.get("score"))
            if score is None:
                score_text = "n/a"
            else:
                score_text = f"{float(score):.4f}"
            preview = (chunk.get("text") or chunk.get("page_content") or "").replace("\n", " ")[:160]
            print(f"    {idx}. score={score_text} | source={source} | {preview}")

    def _print_reranked_top_chunks(self, chunks):
        ranked = sorted(chunks, key=lambda item: float(item.get("rerank_score", item.get("score", 0)) or 0), reverse=True)
        top_chunks = ranked[:5]
        print("  [TOP-5 CHUNKS SAU RERANK]")
        if not top_chunks:
            print("    (không có chunk nào)")
            return

        for idx, chunk in enumerate(top_chunks, 1):
            meta = chunk.get("metadata") or {}
            source = meta.get("source") or meta.get("law_name") or meta.get("document_id") or "Unknown"
            chunk_id = meta.get("chunk_id") or chunk.get("id") or "Unknown"
            score = chunk.get("rerank_score", chunk.get("score"))
            score_text = f"{float(score):.4f}" if score is not None else "n/a"
            preview = (chunk.get("text") or chunk.get("page_content") or "").replace("\n", " ")[:180]
            print(f"    {idx}. rerank_score={score_text} | chunk_id={chunk_id} | source={source} | {preview}")

    def _format_chunks(self, raw_chunks):
        formatted = []
        for chunk in raw_chunks:
            parent = chunk.get("parent") or {}
            doc_text = parent.get("text") or chunk.get("text") or ""

            metadata = (chunk.get("metadata") or {}).copy()
            metadata["chunk_id"] = chunk.get("id", "Unknown")
            metadata["source"] = metadata.get("law_name") or metadata.get("document_id") or "Không xác định nguồn"

            formatted.append(
                {
                    "page_content": doc_text,
                    "text": doc_text,
                    "metadata": metadata,
                    "score": chunk.get("score"),
                }
            )

        return formatted

    def run_query(self, query_text: str):
        print(f"\n{'=' * 72}")
        print(f"[USER] {query_text}")
        print(f"{'=' * 72}")

        started = time.time()

        print("\n[STEP 1] GUARDIAN")
        step_started = time.time()
        guard_res = self.guardian.process_query(query_text)
        status = guard_res.get("status", "unknown")
        intent_info = guard_res.get("intent") or {}
        intent_label = intent_info.get("intent", "unknown")
        reasoning = guard_res.get("reasoning", "")
        response = guard_res.get("response", "")
        print(f"  Tốn: {time.time() - step_started:.2f}s")
        print(f"  Ý định: {intent_label} | Trạng thái: {status}")

        if status != "safe" or guard_res.get("action") == "reject_or_refuse":
            print(f"\n[GUARDIAN BLOCK] {response or reasoning}")
            return

        if intent_label == "out_of_scope":
            print(f"\n[OUT OF SCOPE] {response or reasoning}")
            return

        print("\n[STEP 2] QUERY PROCESSOR")
        step_started = time.time()
        try:
            expanded_queries = self.query_processor.generate_variants(query_text)
        except Exception as exc:
            print(f"  [!] QueryProcessor lỗi: {exc}")
            expanded_queries = [query_text]

        if not expanded_queries:
            expanded_queries = [query_text]

        print(f"  Tốn: {time.time() - step_started:.2f}s")
        print(f"  Mở rộng thành {len(expanded_queries)} truy vấn:")
        for idx, item in enumerate(expanded_queries):
            print(f"    {idx}. {item}")

        print("\n[STEP 3] RETRIEVAL")
        step_started = time.time()
        unique_docs = {}
        for expanded_query in expanded_queries:
            try:
                query_vector = self.embedder.embed_query(expanded_query)
                results = self.indexer.query_with_parent_context(
                    query_vector=query_vector,
                    n_results=self.fusion_top_k,
                    min_score=self.relevance_threshold,
                )
                for result in results:
                    doc_id = result.get("id")
                    if doc_id and doc_id not in unique_docs:
                        unique_docs[doc_id] = result
            except Exception as exc:
                print(f"  [!] Lỗi khi retrieve '{expanded_query}': {exc}")

        raw_chunks = list(unique_docs.values())
        print(f"  Tốn: {time.time() - step_started:.2f}s")
        print(f"  Quét trúng tổng {len(raw_chunks)} đoạn luật gốc hợp lệ.")

        if not raw_chunks:
            fallback = response or "Tôi chưa tìm thấy quy định pháp luật cụ thể trong cơ sở dữ liệu hiện tại."
            print("\n[NO RETRIEVAL RESULTS] Chuyển sang fallback.")
            self._print_answer(fallback, time.time() - started)
            return

        formatted_chunks = self._format_chunks(raw_chunks)
        self._print_chunks(formatted_chunks)

        print("\n[STEP 4] RERANKER")
        step_started = time.time()
        reranked_chunks = formatted_chunks[:self.fusion_top_k]
        if self.use_reranker and formatted_chunks:
            try:
                reranked_result = self.reranker.rerank(query_text, formatted_chunks)
                reranked_chunks = reranked_result[:self.fusion_top_k]
                print(f"  Tốn: {time.time() - step_started:.2f}s")
                print(f"  Chọn lọc ra TOP {len(reranked_chunks)} documents sát nghĩa nhất.")
                if not reranked_chunks:
                    print("  [!] Reranker trả về 0 chunks -> fallback sang retrieve gốc.")
                    reranked_chunks = formatted_chunks[:self.fusion_top_k]
            except Exception as exc:
                print(f"  [!] Lỗi rerank: {exc}")
                reranked_chunks = formatted_chunks[:self.fusion_top_k]
        else:
            print("  [!] Bỏ qua reranker -> dùng trực tiếp chunks đã retrieve.")

        self._print_reranked_top_chunks(reranked_chunks)
        print(f"  └─ Context đưa vào generation: {len(reranked_chunks)} chunks.")

        print("\n[STEP 5] GENERATION")
        step_started = time.time()
        try:
            final_answer = generate_answer(query_text, reranked_chunks)
        except Exception as exc:
            print(f"  [!] Lỗi generation: {exc}")
            final_answer = "Xin lỗi, hệ thống sinh câu trả lời đang gặp sự cố."

        print(f"  Tốn: {time.time() - step_started:.2f}s")
        self._print_answer(final_answer, time.time() - started)

    def _print_answer(self, answer: str, elapsed: float):
        print(f"\n{'=' * 25} TRẢ LỜI CỦA AI {'=' * 25}")
        print(f" Tổng thời gian chạy: {elapsed:.2f}s")
        print("-" * 65)
        print(f"\n{answer}\n")
        print("=" * 66)


def main():
    pipeline = InteractiveRetrievalTest()

    print("\nNhập câu hỏi pháp lý. Gõ 'quit', 'exit', hoặc 'q' để thoát.\n")
    while True:
        try:
            user_query = input("[BẠN]: ").strip()
            if not user_query:
                continue
            if user_query.lower() in {"quit", "exit", "q"}:
                print("Tạm biệt!")
                break

            pipeline.run_query(user_query)
        except KeyboardInterrupt:
            print("\nĐã thoát.")
            break
        except Exception as exc:
            print(f"\n[LỖI]: {exc}")


if __name__ == "__main__":
    main()