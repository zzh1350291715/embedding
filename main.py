from fastapi import FastAPI
from pydantic import BaseModel
from vector_store import insert_document, search_similar_texts
from qwen_llm import call_qwen

app = FastAPI()

class FileRequest(BaseModel):
    knowledge_base_id: int
    file_name: str

class AskRequest(BaseModel):
    knowledge_base_id: int
    query: str

@app.post("/api/knowledge-base/create")
def create_vectors(req: FileRequest):
    return insert_document(req.knowledge_base_id, req.file_name)

@app.post("/api/knowledge-base/ask")
def ask_question(req: AskRequest):
    collection_name = f"knowledge_{req.knowledge_base_id}"
    docs = search_similar_texts(collection_name, req.query)
    context = "\n".join(docs)
    prompt = f"请根据以下资料回答问题：\n{context}\n\n问题：{req.query}\n回答："
    answer = call_qwen(prompt)
    return {"answer": answer}
