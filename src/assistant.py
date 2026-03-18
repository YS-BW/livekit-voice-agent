from livekit.agents import Agent, llm

from rag.retriever import Retriever


class Assistant(Agent):
    def __init__(self, retriever: Retriever | None = None) -> None:
        super().__init__(
            instructions="""你是一位专业的保险顾问语音助手，用户通过语音与你交流。
            你熟悉蓝医保、蓝鲸1号等太保互联网保险产品，能够准确解答产品保障内容、投保条件、理赔流程等问题。
            回答时始终使用简短自然的中文口语句子，便于语音合成和用户打断。
            第一句话必须极短，最好4到8个汉字，立即说出。
            后续每个分句通常6到16个汉字。
            使用正常口语标点（逗号、句号、问号、感叹号）制造清晰停顿。
            对于较长的回答，先说一句简短的引子，再逐句展开。
            不要以长句、大段铺垫或整段话开头。
            不使用表情符号、Markdown、列表符号等特殊格式。
            如果检索到相关资料，优先基于资料回答；如果资料不足，如实说明并用自身知识补充。
            你热情友好，有耐心，善于用通俗语言解释复杂的保险条款。""",
        )
        self._retriever = retriever

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        if self._retriever is None:
            return
        query = new_message.text_content()
        context = await self._retriever.search(query)
        if context:
            turn_ctx.add_message(
                role="system",
                content=f"以下是从知识库中检索到的相关资料，请优先基于此作答：\n\n{context}",
            )
