import logging
from typing import List, Dict, Any
from app.generation.llm_client import chat_completion
from app.utils.config import settings
from app.retrieval.context_assembler import ContextAssembler

logger = logging.getLogger(__name__)

def build_system_prompt() -> str:
    """Build standard system prompt."""
    return settings.get_prompt(
        "legal_rag_generation",
        settings.get_prompt(
            "rag_generator_system",
            "Bạn là trợ lý pháp y chuẩn. Hãy dùng thẻ context để trả lời."
        )
    )

def generate_answer(query: str, retrieved_chunks: List[Any]) -> str:
    """Lấy kết quả Generation RAG từ LLM Client."""
    try:
        gen_config = settings.resolve_ref("models_config.runtime.generator", default={})
    except AttributeError:
        gen_config = {}

    provider = gen_config.get("provider", "openrouter")
    model_id = gen_config.get("model_id", "google/gemini-2.5-flash-lite")
    temperature = gen_config.get("temperature", 0.1)
    max_tokens = gen_config.get("max_tokens", 1536)

    fallback_provider = gen_config.get("fallback_provider")
    fallback_model = gen_config.get("fallback_model")

    assembler = ContextAssembler(max_tokens=max_tokens)
    context_text = assembler.assemble(retrieved_chunks, char_limit=8000)

    user_prompt = f"""Dựa vào các ngữ cảnh pháp lý sau đây, hãy trả lời câu hỏi của tôi:
<Ngữ cảnh>
{context_text}
</Ngữ cảnh>

<Câu hỏi>
{query}
</Câu hỏi>"""

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_prompt}
    ]

    try:
        logger.info(f"Đang yêu cầu LLM {model_id} ({provider}) sinh câu trả lời RAG.")
        response = chat_completion(
            provider=provider,
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.get("text", "")
    except Exception as e:
        logger.error(f"[GENERATOR] Lỗi mô hình ({provider}/{model_id}): {e}")
        if fallback_provider and fallback_model:
            logger.info(f"Chuyển sang Fallback: {fallback_model} via {fallback_provider}")
            try:
                fallback_response = chat_completion(
                    provider=fallback_provider,
                    model_id=fallback_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return fallback_response.get("text", "")
            except Exception as fb_err:
                logger.error(f"[GENERATION] Lỗi Fallback: {fb_err}", exc_info=True)
                return "Xin lỗi, hệ thống bị lỗi kết nối ở cả 2 đầu API."
        else:
            return "Hệ thống AI xử lý ngôn ngữ đang bảo trì."
