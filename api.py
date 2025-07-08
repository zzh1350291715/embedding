from fastapi import APIRouter, UploadFile, File, Body
from fastapi.responses import JSONResponse
from models import FileRequest, AskRequest, GenerateRequest, ChatRequest
from vector_store import insert_document, search_similar_texts
from service import generate_content_from_kb
from qwen_llm import call_qwen, call_qwen_vl
import base64

router = APIRouter()

# ==================== 知识库相关接口 ====================
@router.post("/knowledge-base/create")
def create_vectors(req: FileRequest):
    return insert_document(req.knowledge_base_id, req.file_name)

@router.post("/knowledge-base/ask")
def ask_question(req: AskRequest):
    collection_name = f"knowledge_{req.knowledge_base_id}"
    docs = search_similar_texts(collection_name, req.query)
    context = "\n".join(docs)

    prompt = (
        f"请根据以下资料回答问题：\n{context}\n\n"
        f"请仅使用纯文本作答，不要使用任何 Markdown、符号标记或排版格式，例如 *、#、-、```、** 等。\n\n"
        f"问题：{req.query}\n"
        f"回答："
    )

    answer = call_qwen(prompt)
    return {"answer": answer.strip()}

@router.post("/knowledge-base/generate-outline")
def generate_outline(req: GenerateRequest):
    params = {"estimated_hours": req.estimated_hours}
    return generate_content_from_kb(req.knowledge_base_id, mode="outline", params=params)

@router.post("/knowledge-base/generate-ppt")
def generate_ppt(req: GenerateRequest):
    params = {"chapter_name": req.chapter_name}
    return generate_content_from_kb(req.knowledge_base_id, mode="ppt", params=params)

@router.post("/knowledge-base/generate-quiz")
def generate_quiz(req: GenerateRequest):
    params = {"question_count": req.question_count}
    return generate_content_from_kb(req.knowledge_base_id, mode="quiz", params=params)

# ==================== 文本 AI 对话 ====================
@router.post("/ai-chat/ask")
def ai_chat(req: ChatRequest):
    prompt = (
        f"请以严谨、学术的语气，用清晰准确的语言回答下列问题。\n"
        f"回答需逻辑严密、用词规范，避免使用网络用语或口语表达。\n"
        f"禁止使用 Markdown 或特殊排版符号，如 *, #, -, ** 等。\n\n"
        f"问题：{req.query}\n"
        f"学术性回答："
    )
    answer = call_qwen(prompt)
    return {"answer": answer.strip()}

# ==================== 图像问答接口（Qwen-VL） ====================
@router.post("/ai-chat/image-question")
async def image_question(file: UploadFile = File(...), question: str = Body(...)):
    try:
        # 读取图片并转为 base64
        image_bytes = await file.read()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        base64_url = f"data:{file.content_type};base64,{base64_str}"

        # 调用 Qwen-VL 模型处理
        answer = call_qwen_vl(base64_url, question)
        return {"answer": answer.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
