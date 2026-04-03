from fastapi import APIRouter, Depends, HTTPException
import logging

from app.api.models import ChatRequest, ChatResponse
from app.api.dependencies import get_rag_pipeline
from app.pipeline import LegalRAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Chat"])

@router.get("/health")
def health_check():
    """Kiểm tra trạng thái server"""
    return {"status": "ok", "message": "Legal RAG API is running"}

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest, 
    pipeline: LegalRAGPipeline = Depends(get_rag_pipeline)
):
    """
    Endpoint chat với bot pháp luật.
    Nhận vào câu hỏi (query) và lịch sử trò chuyện (history).
    Trả về câu trả lời kèm các đoạn trích dẫn pháp lý.
    """
    try:
        # Ở đây chúng ta tạm thời có thể truyền luôn query hiện tại vào pipeline
        # (Chưa xử lý history trong đợt này nếu pipeline gốc chưa thiết kế)
        result_dict = pipeline.run(request.query)
        
        return ChatResponse(
            answer=result_dict["answer"],
            sources=result_dict["sources"],
            status=result_dict["status"],
            processing_time=result_dict["processing_time"],
            metadata=result_dict.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
