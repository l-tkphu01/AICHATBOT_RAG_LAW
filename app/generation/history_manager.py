import logging
from typing import List, Dict, Any, Optional
from app.utils.config import settings
from app.generation.llm_client import chat_completion

logger = logging.getLogger(__name__)

class HistoryManager:
    """Mảnh ghép quản lý Lịch sử Chat - Tối ưu Context cho LLM"""
    
    def __init__(self):
        # Lấy giá trị cấu hình từ history_config.yaml
        self.history_config = getattr(settings, "history", {})
        self.context_config = self.history_config.get("context", {})
        
        self.max_messages = self.context_config.get("max_messages_per_prompt", 15)
        self.summarize_after = self.context_config.get("summarize_after_messages", 10)
        self.max_tokens = self.context_config.get("max_tokens_for_history", 2200)
        
    def _approximate_tokens(self, text: str) -> int:
        """Ước lượng số token của văn bản tiếng Việt (~3 ký tự/token)"""
        return len(text) // 3

    def process_history(self, history: List[Any]) -> str:
        """
        Xử lý mảng raw history từ Frontend thành chuỗi tối ưu đem nhét vào Prompt.
        Áp dụng chiến lược kiểm tra vượt ngưỡng -> tóm tắt/cắt gọt.
        """
        if not history:
            return ""

        # Chuẩn hóa về list dictionary
        norm_hist = []
        for msg in history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                norm_hist.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, dict):
                norm_hist.append({"role": msg.get("role", ""), "content": msg.get("content", "")})

        if not norm_hist:
            return ""

        total_tokens = sum(self._approximate_tokens(m["content"]) for m in norm_hist)
        
        # Nếu vượt quá số message hoặc vượt ngưỡng Token -> Gọi kỹ thuật Hybrid Summarize
        if len(norm_hist) > self.summarize_after or total_tokens > self.max_tokens:
            return self._hybrid_summarize(norm_hist)
        
        # Nếu an toàn, nối lại và trả về nguyên bản
        return self._format_as_text(norm_hist)

    def _hybrid_summarize(self, history: List[Dict[str, str]]) -> str:
        """
        Kỹ thuật Hybrid:
        - Tóm tắt khúc đầu (phần cũ nhất).
        - Giữ nguyên vẹn 4 tin nhắn khúc cuối để AI hiểu rõ chi tiết hiện tại.
        """
        keep_recent = 4
        if len(history) <= keep_recent:
            return self._format_as_text(history)
            
        old_msgs = history[:-keep_recent]
        recent_msgs = history[-keep_recent:]
        
        old_text = self._format_as_text(old_msgs)
        summary = self._call_summarize_llm(old_text)
        
        final_text = f"[Tóm tắt bối cảnh cũ]:\n{summary}\n\n[Tin nhắn gần đây]:\n{self._format_as_text(recent_msgs)}"
        return final_text

    def _call_summarize_llm(self, text: str) -> str:
        """Gọi LLM (nhẹ) để tóm tắt các đoạn hội thoại cũ"""
        try:
            logger.info("Bắt đầu tóm tắt lịch sử chat do vượt ngưỡng (Strategy: Hybrid)...")
            
            # Lấy model summarizer từ settings (vốn dùng cho tóm tắt input dài)
            model_cfg = settings.resolve_ref("models_config.runtime.input_summarizer", default={})
            provider = model_cfg.get("provider", "groq")
            model_id = model_cfg.get("model_id", "llama-3.1-8b-instant")
            
            prompt = f"Hãy đọc đoạn hội thoại sau và tóm tắt ngắn gọn các sự kiện/tình huống pháp lý chính của người dùng trong khoảng 2-3 câu:\n\n{text}"
            
            resp = chat_completion(
                provider=provider,
                model_id=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Text tóm tắt cần sự chính xác cao, nhiệt độ thấp
                max_tokens=256
            )
            
            content = str(resp.get("text", "")).strip()
            if not content:
                content = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
            return content if content else "[Không thể tóm tắt lịch sử]"
        except Exception as e:
            logger.error(f"Lỗi khi tóm tắt lịch sử: {e}")
            # Fallback: ngắt chữ cứng thay vì để sập hệ thống
            return text[:400] + " ... [Lịch sử cũ đã bị cắt đi do lỗi]"

    def _format_as_text(self, history: List[Dict[str, str]]) -> str:
        """Format chuỗi hiển thị theo Khách / Bot"""
        lines = []
        for msg in history:
            role = "Khách" if msg["role"] == "user" else "Luật sư AI"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)