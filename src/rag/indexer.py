"""
离线索引构建脚本：读取 docs/ 目录下的 MD/TXT/PDF 文件，
使用 DashScope text-embedding-v3 生成向量并持久化到 data/rag_index/。

用法：
    uv run src/rag/indexer.py
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag.indexer")

DOCS_DIR = Path("docs")
INDEX_DIR = Path("data/rag_index")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"


def build_index() -> None:
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding

    if not DASHSCOPE_API_KEY:
        logger.error("DASHSCOPE_API_KEY 未设置，请检查 .env.local")
        sys.exit(1)

    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        logger.error("docs/ 目录为空或不存在，请先放入文档")
        sys.exit(1)

    embed_model = OpenAIEmbedding(
        model=EMBED_MODEL,
        api_base=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
    )
    Settings.embed_model = embed_model
    Settings.llm = None  # 构建索引不需要 LLM

    logger.info("加载文档：%s", DOCS_DIR)
    documents = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        recursive=True,
        required_exts=[".md", ".txt", ".pdf"],
        filename_as_id=True,
    ).load_data()
    logger.info("共加载 %d 个文档片段", len(documents))

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("构建向量索引，保存到 %s …", INDEX_DIR)
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    logger.info("索引构建完成")


if __name__ == "__main__":
    build_index()
