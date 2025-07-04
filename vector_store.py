import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)

# 初始化 Milvus 连接
connections.connect("default", host="localhost", port="19530")


def insert_document(knowledge_base_id: int, file_name: str):
    file_path = os.path.join("uploads", str(knowledge_base_id), file_name)

    if not os.path.exists(file_path):
        return {"error": f"file not found: {file_path}"}

    # 1. 加载文档
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()

    # 2. 文本分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # 3. 向量化
    texts = [doc.page_content for doc in split_docs]
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = embed.embed_documents(texts)

    collection_name = f"knowledge_{knowledge_base_id}"
    if collection_name not in list_collections():
        create_collection(collection_name)

    # 4. 插入数据（三列：id + embedding + text）
    collection = Collection(collection_name)
    ids = list(range(len(vectors)))
    insert_data = [ids, vectors, texts]
    collection.insert(insert_data)
    collection.flush()

    # 5. 创建索引（embedding 字段）
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
    )

    # 6. 加载 collection（否则 Attu 无法访问）
    collection.load()

    return {"msg": f"{len(vectors)} 条向量与文本已插入并加载到集合 {collection_name}"}


def create_collection(name: str):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="Knowledge base collection")
    Collection(name, schema)


def list_collections():
    return utility.list_collections()
