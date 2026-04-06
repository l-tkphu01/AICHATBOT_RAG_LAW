import os
import sys # Thao tác với môi trường python, thêm thư mục vào sys.path để import module từ project

# Add root project dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import LegalRAGPipeline


def print_result(res): # res là kết quả trả về từ pipeline, có thể chứa nhiều thông tin như status, answer, llm_used, v.v. trả về dưới dạng dictionary
    status = res.get("status", "unknown")
    answer = res.get("answer", "")

    if status in ["blocked", "reject_or_refuse", "unknown"]:
        print(f"🛑 Trạng thái Guardian: {status}")
        print(f"🤖 Bot (Random Tĩnh): {answer}")
    elif status in ["out_of_scope", "unclear_intent", "chitchat", "clarify", "uncertain", "unknown"]:
        print(f"⚠️ Trạng thái Guardian: {status}")
        print(f"🤖 Bot (Groq 8B Soft Refusal): {answer}")
    elif status == "success":
        print(f"✅ Trạng thái Guardian: {status}")
        if len(answer) > 500:
            print(f"🤖 Bot (Gemma 31B RAG trả lời):\n{answer[:500]}...")
        else:
            print(f"🤖 Bot (Gemma 31B RAG trả lời): {answer}")
    else:
        print(f"Trạng thái: {status}")
        print(f"Đáp án: {answer}")


def main():
    print("Khởi tạo hệ thống Legal RAG Pipeline (chế độ nhập tay)...")
    pipeline = LegalRAGPipeline()
 
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU KIỂM TRA HỆ THỐNG TỪ CHỐI & PHÂN LUỒNG LLM (CHẾ ĐỘ THỦ CÔNG) 🚀")
    print("Nhập 'q', 'quit' hoặc 'exit' để thoát.")
    print("="*80 + "\n")

    while True:
        query = input("👤 User: ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            print("👋 Đã thoát chế độ test thủ công.")
            break

        if not query:
            print("Vui lòng nhập câu hỏi hoặc gõ 'q' để thoát.")
            print("-" * 80)
            continue

        try:
            res = pipeline.run(query)
            print_result(res)
        except Exception as e:
            print(f"❌ LỖI: {e}")

        print("-" * 80)

if __name__ == "__main__":
    main()
