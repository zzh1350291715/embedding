from fastapi import FastAPI
from pydantic import BaseModel
from vector_store import insert_document

app = FastAPI()

class FileRequest(BaseModel):
    knowledge_base_id: int
    file_name: str

@app.post("/api/knowledge-base/create")
def create_vectors(req: FileRequest):
    return insert_document(req.knowledge_base_id, req.file_name)
