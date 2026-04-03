# app/ingestion/__init__.py
"""
Package ingestion — Tập hợp các thành phần của ingestion pipeline.

Luồng xử lý:
    PDF → LegalPDFProcessor → LegalChunker → EmbeddingGenerator → LegalIndexer

Dùng nhanh qua IngestionPipeline:
    from app.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline.run("data/raw_pdfs/luat_dat_dai.pdf", law_name="Luật Đất đai 2024")
"""

from app.ingestion.pdf_processor import (
    LegalPDFProcessor,
    LegalStructure,
    RawBlock,
)

from app.ingestion.legal_chunker import (
    LegalChunker,
    LegalChunk,
)

from app.ingestion.embedder import (
    EmbeddingGenerator,
)

from app.ingestion.indexer import (
    LegalIndexer,
)

from app.ingestion.run_ingestion import (
    IngestionPipeline,
)

__all__ = [
    # PDF parsing / structure extraction
    "LegalPDFProcessor",
    "LegalStructure",
    "RawBlock",
    # Chunking
    "LegalChunker",
    "LegalChunk",
    # Embedding
    "EmbeddingGenerator",
    # Storage / retrieval index
    "LegalIndexer",
    # Orchestrator
    "IngestionPipeline",
]
