import json
from typing import List, Dict, Any, Optional

class ContextAssembler:
    """
    Module for assembling and truncating context chunks for LLM generation.
    """

    def __init__(self, max_tokens: int = 4000) -> None:
        self.max_tokens = max_tokens

    def assemble(self, scored_chunks: List[Any], char_limit: int = 8000) -> str:
        """
        Assemble the text from scored chunks, preserving citation metadata,
        and truncating the output to char_limit to fit the LLM context window.
        """
        assembled_context = ""
        current_length = 0
        
        if not scored_chunks:
            return "Không tìm thấy dữ liệu quy định tham khảo."

        for i, doc in enumerate(scored_chunks, start=1):
            if isinstance(doc, dict):
                content = doc.get("page_content", doc.get("content", doc.get("text", str(doc))))
                metadata = doc.get("metadata", {})
            else:
                content = getattr(doc, "page_content", getattr(doc, "content", getattr(doc, "text", str(doc))))
                metadata = getattr(doc, "metadata", {})

            source = metadata.get("source", metadata.get("document_id", "Không rõ nguồn"))
            id_chunk = metadata.get("chunk_id", "Không rõ ID")
            
            chunk_block = f"--- [TÀI LIỆU {i}] ---\n"
            chunk_block += f"Nguồn: {source} (Mã phần: {id_chunk})\n"
            chunk_block += f"Nội dung: {content}\n\n"
            
            if current_length + len(chunk_block) > char_limit:
                break
                
            assembled_context += chunk_block
            current_length += len(chunk_block)
            
        return assembled_context.strip()
