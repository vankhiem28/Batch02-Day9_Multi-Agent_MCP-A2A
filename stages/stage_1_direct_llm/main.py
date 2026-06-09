import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from common.llm import get_llm

async def main():
    print("Nhập câu hỏi về dev:")
    question=input()
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "Bạn là một senior dev với 10 năm kinh nghiệm"
                "Trả lời ngắn ngọn, dễ hiểu"
                "Trình bày format câu hỏi dễ hiểu"
            )
        ),
        HumanMessage(content=question),
    ]

    print("\n>>> Calling LLM directly (no tools, no RAG)...\n")
    response = await llm.ainvoke(messages)
    print(response.content)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())