import os
import sys
from dotenv import load_dotenv

# Thêm đường dẫn thư mục gốc vào sys.path để Python hiểu các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.embedder import EmbeddingGenerator
from app.ingestion.indexer import LegalIndexer

def test_retrieval(query_text: str, top_k: int = 5):
    """
    Tìm kiếm và in ra Top K chunk trả về.
    """
    print(f"\n[?] Câu hỏi: '{query_text}'\n")

    # 1. Nhúng vector cho câu hỏi (dùng Cohere như đã cấu hình)
    embedder = EmbeddingGenerator()
    query_vector = embedder.embed_query(query_text)
    
    # 2. Kết nối tới ChromaDB
    indexer = LegalIndexer()
    
    # 3. Tìm kiếm + nối parent context
    # Dùng query_with_parent_context() lấy luôn nội dung Điều luật (nếu có)
    results = indexer.query_with_parent_context(
        query_vector=query_vector,
        n_results=top_k,
        min_score=0.2 # Ngưỡng điểm để lọc kết quả kém
    )
    
    # 4. In kết quả
    if not results:
        print("[!] Không tìm thấy nội dung phù hợp.")
        return

    print(f"--- TÌM THẤY {len(results)} KẾT QUẢ ---")
    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        chunk_text = res.get("text", "")
        meta = res.get("metadata", {})
        
        # In các thông tin
        print(f"\n[{i}] Độ tương đồng (Score): {score:.4f}")
        print(f"    - Nguồn: {meta.get('part', '')} > {meta.get('chapter', '')} > {meta.get('section', '')} > {meta.get('article', '')}")
        print(f"    - Text:")
        # In rút gọn để đỡ rối mắt
        truncated_text = chunk_text[:200].replace('\n', ' ') + "..." if len(chunk_text) > 200 else chunk_text.replace('\n', ' ')
        print(f"      {truncated_text}")
        
if __name__ == "__main__":
    load_dotenv() # Load biến môi trường (COHERE_API_KEY)
    
    # Đọc câu hỏi từ CLI argument (nếu có), nếu không dùng câu hỏi mặc định
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Hồ sơ đăng ký thuế bao gồm những gì?"
        
    test_retrieval(query)
