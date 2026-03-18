"""
在线检索模块：从持久化的 LlamaIndex 索引中检索相关文本段落。
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("rag.retriever")

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"


class Retriever:
    def __init__(self, retriever) -> None:
        self._retriever = retriever

    @classmethod
    def load(cls, index_dir: Path, top_k: int = 3) -> "Retriever | None":
        """加载持久化索引，索引不存在时返回 None（优雅降级）。"""
        docstore_path = index_dir / "docstore.json"
        if not index_dir.exists() or not docstore_path.exists():
            logger.warning("RAG 索引未找到（%s），跳过 RAG 功能", index_dir)
            return None

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("DASHSCOPE_API_KEY 未设置，跳过 RAG 功能")
            return None

        try:
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.embeddings.openai import OpenAIEmbedding

            embed_model = OpenAIEmbedding(
                model=EMBED_MODEL,
                api_base=DASHSCOPE_BASE_URL,
                api_key=api_key,
            )
            Settings.embed_model = embed_model
            Settings.llm = None

            storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
            index = load_index_from_storage(storage_context)
            retriever = index.as_retriever(similarity_top_k=top_k)
            logger.info("RAG 索引加载成功（top_k=%d）", top_k)
            return cls(retriever)
        except Exception:
            logger.exception("加载 RAG 索引失败，跳过 RAG 功能")
            return None

    async def search(self, query: str) -> str:
        """检索与 query 最相关的段落，返回拼接后的纯文本。"""
        if not query.strip():
            return ""
        try:
            nodes = await self._retriever.aretrieve(query)
            if not nodes:
                return ""
            seen: set[str] = set()
            parts: list[str] = []
            for node in nodes:
                text = node.get_content().strip()
                if text and text not in seen:
                    seen.add(text)
                    parts.append(text)
            return "\n\n".join(parts)
        except Exception:
            logger.exception("RAG 检索出错")
            return ""
