from pydantic import BaseModel
from typing import Optional

class FileRequest(BaseModel):
    knowledge_base_id: int
    file_name: str

class AskRequest(BaseModel):
    knowledge_base_id: int
    query: str

class GenerateRequest(BaseModel):
    knowledge_base_id: int
    estimated_hours: Optional[int] = None
    chapter_name: Optional[str] = None
    question_count: Optional[int] = None

class ChatRequest(BaseModel):
    query: str