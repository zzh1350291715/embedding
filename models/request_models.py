from pydantic import BaseModel

class FileRequest(BaseModel):
    file_name: str

class QueryRequest(BaseModel):
    question: str
