import os
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pymilvus import connections, Collection, utility, FieldSchema, DataType, CollectionSchema

connections.connect("default", host="localhost", port="19530")

def list_collections():
    return utility.list_collections()

def create_collection(name: str):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="Knowledge base collection")
    Collection(name, schema)

def get_max_id(collection_name: str) -> int:
    collection = Collection(collection_name)
    if collection.num_entities == 0:
        return -1
    res = collection.query(expr="id >= 0", output_fields=["id"])
    if not res:
        return -1
    return max(item["id"] for item in res)

def get_loader_by_suffix(file_path: str):
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".pdf":
        return PyMuPDFLoader(file_path)
    elif suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif suffix == ".pptx":
        return UnstructuredPowerPointLoader(file_path)
    elif suffix == ".docx":
        return UnstructuredWordDocumentLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path)

def insert_document(knowledge_base_id: int, file_name: str):
    file_path = os.path.join("uploads", str(knowledge_base_id), file_name)
    if not os.path.exists(file_path):
        return {"error": f"file not found: {file_path}"}
    loader = get_loader_by_suffix(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    texts = [doc.page_content for doc in split_docs]
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = embed.embed_documents(texts)
    min_len = min(len(vectors), len(texts))
    texts = texts[:min_len]
    vectors = vectors[:min_len]
    collection_name = f"knowledge_{knowledge_base_id}"
    if collection_name not in list_collections():
        create_collection(collection_name)
    max_id = get_max_id(collection_name)
    ids = list(range(max_id + 1, max_id + 1 + min_len))
    collection = Collection(collection_name)
    collection.insert([ids, vectors, texts])
    collection.flush()
    if not collection.has_index():
        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )
    collection.load()
    return {"msg": f"{min_len} 条向量与文本已插入并加载到集合 {collection_name}"}

def search_similar_texts(collection_name: str, query: str, top_k: int = 5):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_vector = embed.embed_query(query)
    collection = Collection(collection_name)
    collection.load()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    hits = results[0]
    return [hit.entity.get("text") for hit in hits]
