import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, Float, ForeignKey, JSON
from datetime import datetime, timezone

from app.utils.config import settings

logger = logging.getLogger(__name__)

# Đọc cấu trúc kết nối DB từ system_config (nếu có)
db_config = getattr(settings, "database", {})
db_driver = db_config.get("driver", "postgresql+asyncpg")
db_user = db_config.get("user", "chatbot_user")
db_password = db_config.get("password", "chatbot_pass_2026")
db_host = db_config.get("host", "localhost")
db_port = db_config.get("port", 5432)
db_name = db_config.get("name", "chatbot_rag")

DATABASE_URL = f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

try:
    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
except Exception as e:
    logger.error(f"Không thể khởi tạo SQLAlchemy Engine: {e}")
    engine = None
    AsyncSessionLocal = None

Base = declarative_base()

# Lấy tên bảng từ file history_config.yaml (chuẩn hóa Production)
h_config = getattr(settings, "history", {})
tb_sessions = h_config.get("database", {}).get("table_sessions", "chat_sessions")
tb_messages = h_config.get("database", {}).get("table_messages", "chat_messages")

class ChatSession(Base):
    """Mô hình lưu trữ Phiên người dùng (Theo đúng định dạng session management)"""
    __tablename__ = tb_sessions
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    device_id: Mapped[str] = mapped_column(String(255), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages: Mapped[list["ChatMessage"]] = relationship("ChatMessage", back_populates="session", cascade="all, delete")

class ChatMessage(Base):
    """Mô hình lưu trữ Tin nhắn & Metadata (LLM Input/Output Tracker)"""
    __tablename__ = tb_messages
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(255), ForeignKey(f"{tb_sessions}.session_id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(50)) # user, assistant
    content: Mapped[str] = mapped_column(String)
    
    # Metadata tracking (Nắm bắt từ history_config.yaml -> metadata.fields)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)
    legal_sources: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

