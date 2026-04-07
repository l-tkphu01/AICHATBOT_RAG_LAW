import logging
import json
from app.utils.config import settings

logger = logging.getLogger(__name__)

class RedisCache:
    """Tầng Cache sử dụng Redis để lưu lịch sử Fast Layer (Theo history_config.yaml)"""
    def __init__(self):
        self.history_config = getattr(settings, "history", {})
        self.cache_config = self.history_config.get("cache", {})
        
        self.enabled = self.cache_config.get("enabled", True)
        self.ttl = self.cache_config.get("ttl_seconds", 86400)
        self.prefix = self.cache_config.get("prefix_key", "chat_session:")
        self.max_messages = self.cache_config.get("max_messages_in_cache", 20)
        
        self.redis_client = None
        if self.enabled:
            self._connect_redis()

    def _connect_redis(self):
        try:
            # Requires `pip install redis`
            import redis.asyncio as redis
            # TODO: Cấu hình connection host có thể linh hoạt đọc từ file môi trường
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            logger.info("Kết nối Redis Cache thành công.")
        except ImportError:
            logger.warning("Thư viện 'redis' chưa được cài đặt. Bỏ qua cấu hình Cache. Chạy `pip install redis`")
            self.redis_client = None
        except Exception as e:
            logger.error(f"Không thể kết nối Redis: {e}. Fallback to PostgreSQL.")
            self.redis_client = None

    async def get_session(self, session_id: str) -> list:
        if not self.redis_client:
            return []
            
        key = f"{self.prefix}{session_id}"
        try:
            data = await self.redis_client.get(key)
            return json.loads(data) if data else []
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu Redis (Session {session_id}): {e}")
            return []

    async def save_session(self, session_id: str, messages: list):
        if not self.redis_client:
            return
            
        key = f"{self.prefix}{session_id}"
        try:
            # Chỉ lưu n messages gần nhất để chống tràn RAM Redis
            cache_messages = messages[-self.max_messages:]
            await self.redis_client.set(key, json.dumps(cache_messages), ex=self.ttl)
        except Exception as e:
            logger.error(f"Lỗi ghi dữ liệu Redis (Session {session_id}): {e}")

redis_cache = RedisCache()
