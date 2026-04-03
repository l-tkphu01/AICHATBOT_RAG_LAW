import warnings
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as chat_router

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Tắt cảnh báo Pydantic hoặc thư viện cũ nếu có
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Legal RAG API Chatbot",
    description="API cho Chatbot Pháp luật sử dụng kiến trúc RAG, Cohere và OpenRouter",
    version="1.0.0"
)

# Cấu hình CORS (cho phép gọi từ mọi nguồn Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các endpoints
app.include_router(chat_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    logging.info("Starting up Legal RAG API Server...")
    # Việc tiền khởi tạo model / vector db có thể gọi ở đây,
    # nhưng hiện tại đã sử dụng _pipeline_instance với singleton lazy load.
    pass

@app.get("/")
def read_root():
    return {"message": "Welcome to Legal RAG Chatbot API"}
