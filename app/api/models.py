from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict
import html

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the sender (user, assistant)")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    # LỚP 2: Giới hạn độ dài tối đa 2000 ký tự để chống Prompt Injection nhồi nhét
    query: str = Field(..., max_length=2000, description="Phần câu hỏi mới nhất của người dùng")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Lịch sử trò chuyện trước đó")
    stream: bool = Field(False, description="Cờ đánh dấu trả về dạng Stream, bỏ qua nếu chưa hỗ trợ")

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # LỚP 1: Chống XML/HTML Injection bằng cách Escape ký tự
        return html.escape(v)

class SourceChunk(BaseModel):
    chunk_id: str
    source_name: str
    preview_text: str
    score: float

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời từ RAG Pipeline")
    sources: List[SourceChunk] = Field(default_factory=list, description="Các đoạn tài liệu pháp lý trích dẫn")
    status: str = Field("success", description="Trạng thái thực thi (success, blocked, out_of_scope, error)")
    processing_time: float = Field(0.0, description="Thời gian chạy pipeline (giây)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata liên quan đến Guardian, Query Processor")
