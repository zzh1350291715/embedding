from typing import Optional

from vector_store import search_similar_texts
from qwen_llm import call_qwen

def generate_content_from_kb(knowledge_base_id: int, mode: str, params: Optional[dict] = None):
    collection_name = f"knowledge_{knowledge_base_id}"
    context_list = search_similar_texts(collection_name, query="课程主要内容", top_k=10)
    context = "\n".join(context_list)

    if mode == "outline":
        extra = ""
        if params and params.get("estimated_hours") is not None:
            extra = f"预计授课时长：{params['estimated_hours']}小时。\n"
        prompt = f"""请根据以下课程资料内容生成一个完整的授课大纲，包括章节标题和简要描述。
{extra}
请仅输出纯文本，不要包含任何 Markdown 格式符号或其它格式化字符：
{context}
请按照如下格式输出：
1. 第一章 标题：xxxx
   内容概述：...
2. 第二章 标题：xxxx
   内容概述：...
"""
    elif mode == "ppt":
        extra = ""
        if params and params.get("chapter_name"):
            extra = f"请重点围绕章节：{params['chapter_name']}。\n"
        prompt = f"""请根据以下课程内容生成一个授课用的 PPT 提纲，包含每页标题和主要要点。
{extra}
请仅输出纯文本，不要包含任何 Markdown 格式符号或其它格式化字符：
{context}
请输出格式如下：
第1页：标题：...
内容要点：
- xxx
- xxx
"""
    elif mode == "quiz":
      count = params.get("question_count", 5) if params else 5
      extra = f"请生成{count}道考试题目，类型包括选择题、简答题和判断题\n"
      prompt = f"""请根据以下课程内容，{extra}
请仅输出纯文本，不要包含任何 Markdown 格式符号或其它格式化字符：
请按照如下格式输出每一道题：
题目：...
类型：（选择题 / 判断题 / 简答题）
选项：（如为选择题，需提供 A. xxx B. xxx C. xxx D. xxx）
答案：（正确答案或参考答案）
"""
    else:
        return {"error": "未知模式"}

    answer = call_qwen(prompt)
    return {"result": answer.strip()}
