import os
import sys
import json
from dotenv import load_dotenv

# Thêm đường dẫn project vào hệ thống để import module app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from app.guardian.pipeline import GuardianPipeline

def test_limits():
    print("Khởi tạo Guardian Pipeline (Bỏ qua Model Nhúng/ChromaDB)...")
    guardian = GuardianPipeline()
    
    base_text = "Thưa luật sư, tôi có một vụ việc như sau: Mảnh đất nhà tôi do bố mẹ ranh giới từ năm 1990 không có sổ đỏ. Nay xã thu hồi mở đường nhưng bồi thường giá quá thấp. Chủ tịch xã ký quyết định thu hồi có đúng thẩm quyền không? Mức bồi thường đất không sổ đỏ được tính thế nào? " # ~276 ký tự
    
    scenarios = {
        "1_NORMAL (Dưới 1000)": base_text,  # Khoảng 276 ký tự -> Cho qua, tìm kiếm RAG bình thường
        "2_SUMMARIZE_ONLY (1000 - 1800)": (base_text * 5) + " Mong luật sư tóm tắt và giải đáp giúp.", # ~1400 ký tự -> Gọi Llama 8B tóm tắt
        "3_WARNING_AND_SUM_ (1800 - 2000)": (base_text * 7), # ~1932 ký tự -> Gọi Llama 8B tóm tắt + Ném cảnh báo (Warning)
        "4_REJECT_MAX_CHARS (Trên 2000)": (base_text * 10), # ~2760 ký tự -> Đạp thẳng, Reject
        "5_REJECT_MAX_LINES (Nhồi Enter)": "Chào luật sư.\n" * 35 # 35 dòng Enter -> Đạp thẳng, Reject
    }
    
    for name, query in scenarios.items():
        line_cnt = len(query.split('\n'))
        char_cnt = len(query)
        print(f"\n=======================================================")
        print(f"BÀI TEST: {name} | Ký tự: {char_cnt} | Số dòng: {line_cnt}")
        print(f"=======================================================")
        
        # Chạy query qua Guardian thay vì toàn bộ Pipeline
        result = guardian.process_query(query)
        
        print(f"-> TRẠNG THÁI (STATUS): {result.get('status')}")
        print(f"-> HÀNH ĐỘNG (ACTION): {result.get('action')}")
        
        if "warning" in result:
             print(f"-> CẢNH BÁO (WARNING): {result.get('warning')}")
             
        print(f"-> LÝ DO (REASONING): {result.get('reasoning', '')}")
        
        if result.get("response"):
            ans = result.get('response')
            print(f"-> TRẢ LỜI CỦA BOT: {ans[:150]}..." if len(ans) > 150 else f"-> TRẢ LỜI CỦA BOT: {ans}")

if __name__ == '__main__':
    test_limits()