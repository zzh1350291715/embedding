from pydantic import BaseModel
from vector_store import insert_document, search_similar_texts
from qwen_llm import call_qwen
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = [
    "http://localhost:8080",     # 你的前端地址
    "http://127.0.0.1:8080"
]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # 指定允许的源
    allow_credentials=True,
    allow_methods=["*"],                # 允许所有请求方法，如 POST, GET
    allow_headers=["*"],                # 允许所有请求头
)

class FileRequest(BaseModel):
    knowledge_base_id: int
    file_name: str

class AskRequest(BaseModel):
    knowledge_base_id: int
    query: str

@app.post("/api/knowledge-base/create")
def create_vectors(req: FileRequest):
    return insert_document(req.course_id, req.file_name)

@app.post("/api/knowledge-base/ask")
def ask_question(req: AskRequest):
    collection_name = f"knowledge_{req.knowledge_base_id}"
    docs = search_similar_texts(collection_name, req.query)
    context = "\n".join(docs)
    prompt = f"请根据以下资料回答问题：\n{context}\n\n问题：{req.query}\n回答："
    answer = call_qwen(prompt)
    return {"answer": answer}
