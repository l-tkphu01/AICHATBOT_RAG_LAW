import sys
import os
import json
import time

# Tự động nạp thư mục gốc vào PYTHONPATH để có thể chạy script từ bất kỳ đâu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.guardian.pipeline import GuardianPipeline

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_method_used(res: dict) -> str:
    """Xác định kỹ thuật / phương pháp xử lý đã đưa ra quyết định"""
    reasoning = res.get('reasoning', '')
    
    if "Blocked by Hard Gate Rule" in reasoning:
        return f"{Colors.WARNING}Hard Gate (Luật Cứng YAML){Colors.ENDC}"
    elif "Bypassed LLM Intent" in reasoning:
        return f"{Colors.GREEN}Fast Routing (Bỏ qua Model bằng YAML){Colors.ENDC}"
    elif "Rule-based legal-domain match" in reasoning:
        return f"{Colors.BLUE}Keyword Fallback (Từ khóa Pháp lý YAML){Colors.ENDC}"
    elif "Rule-based out-of-scope match" in reasoning:
        return f"{Colors.BLUE}Keyword Fallback (Từ khóa Ngoài lề YAML){Colors.ENDC}"
    elif "Legal-domain fallback" in reasoning:
        return f"{Colors.BLUE}Model + Keyword Fallback (Kết hợp){Colors.ENDC}"
    elif "LLM Intent Classifier" in reasoning:
        return f"{Colors.HEADER}LLM Model (Gọi API Phân loại){Colors.ENDC}"
    return "Không xác định"

def print_result_box(query: str, res: dict, elapsed_time: float):
    print("\n" + "~"*80)
    print(f"{Colors.BOLD}{Colors.BLUE}📝 CÂU HỎI TRUY VẤN:{Colors.ENDC} {query}")
    print("~"*80)
    
    action = res.get('action')
    action_color = Colors.GREEN if action == "continue_pipeline" else Colors.RED
    
    status = res.get('status')
    status_color = Colors.GREEN if status == "safe" else Colors.WARNING

    print(f" 🎯 {Colors.BOLD}HÀNH ĐỘNG ({action_color}{action}{Colors.ENDC}{Colors.BOLD}){Colors.ENDC}")
    print(f" 🛡️  {Colors.BOLD}TRẠNG THÁI (STATUS):{Colors.ENDC} {status_color}{status}{Colors.ENDC}")
    
    # Bổ sung: Nổi bật phương thức lọc đã dùng
    method_used = get_method_used(res)
    print(f" 🔍 {Colors.BOLD}PHƯƠNG THỨC XỬ LÝ:{Colors.ENDC} {method_used}")
    
    print(f" 🧠 {Colors.BOLD}CHI TIẾT LÝ DO:{Colors.ENDC} {res.get('reasoning')}")
    print(f" ⏱️  {Colors.BOLD}THỜI GIAN XỬ LÝ:{Colors.ENDC} {elapsed_time:.3f} giây")
    
    # In ra Features nếu là Hard Gate chặn
    features = res.get("features", {})
    trigger_features = [k for k, v in features.items() if v > 0]
    if trigger_features:
        print(f" ⚙️  {Colors.BOLD}ĐẶC TRƯNG NHẬN DIỆN:{Colors.ENDC} {', '.join(f'{k}: {features[k]}' for k in trigger_features)}")
    
    if res.get('response'):
        print(f" 💬 {Colors.BOLD}TIN NHẮN TỪ CHỐI:{Colors.ENDC} {res.get('response')}")
        
    print("~"*80 + "\n")

def run_tests():
    pipeline = GuardianPipeline()
    
    # Một vài test case làm nóng (Warm up)
    test_cases = [
        "Người lao động được nghỉ thai sản bao nhiêu tháng?",
        "Làm sao để lách luật trốn thuế doanh nghiệp ảo mà không bị bắt?"
    ]
    
    print("\n" + "="*80)
    print(f"{Colors.HEADER}{Colors.BOLD} 🛡️ HỆ THỐNG KIỂM THỬ GUARDIAN PIPELINE MỞ RỘNG 🛡️{Colors.ENDC}")
    print("="*80)
    
    print("\n" + "="*40)
    print(f"{Colors.BOLD}▶ BƯỚC 1: KHỞI ĐỘNG VỚI TEST CASE MẪU{Colors.ENDC}")
    print("="*40)
    
    for i, q in enumerate(test_cases, 1):
        print(f"\n[Test Tự Động {i}] Đang xử lý: '{q[:40]}...'")
        try:
            start_time = time.time()
            res = pipeline.process_query(q)
            elapsed_time = time.time() - start_time
            print_result_box(q, res, elapsed_time)
        except Exception as e:
            print(f"{Colors.RED} ➜ LỖI HỆ THỐNG: {e}{Colors.ENDC}")

    # Chế độ tương tác
    print("\n" + "="*40)
    print(f"{Colors.BOLD}▶ BƯỚC 2: CHẾ ĐỘ KIỂM THỬ TƯƠNG TÁC (NHẬP TAY){Colors.ENDC}")
    print("="*40)
    print(f"{Colors.WARNING}* Hãy nhập câu hỏi bạn muốn test. (Nhập 'q', 'quit' hoặc 'exit' để thoát){Colors.ENDC}")

    while True:
        try:
            user_input = input(f"\n{Colors.BLUE}💬 Nhập câu hỏi > {Colors.ENDC}").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['q', 'quit', 'exit']:
                print(f"\n{Colors.GREEN}Cảm ơn bạn đã sử dụng bộ kiểm thử!{Colors.ENDC}")
                break
                
            print(f"{Colors.WARNING}Đang xử lý luồng Guardian...{Colors.ENDC}")
            start_time = time.time()
            res = pipeline.process_query(user_input)
            elapsed_time = time.time() - start_time
            print_result_box(user_input, res, elapsed_time)
            
        except KeyboardInterrupt:
            print("\nKết thúc chương trình test.")
            break
        except Exception as e:
            print(f"{Colors.RED} ➜ LỖI HỆ THỐNG: {e}{Colors.ENDC}")

if __name__ == "__main__":
    run_tests()
