import logging
from app.pipeline import LegalRAGPipeline

logger = logging.getLogger(__name__)
_pipeline_instance = None

def get_rag_pipeline() -> LegalRAGPipeline:
    """Dependency injection to get the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing LegalRAGPipeline for dependency injection...")
        _pipeline_instance = LegalRAGPipeline.get_instance()
    return _pipeline_instance
