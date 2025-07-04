from pymilvus import Collection
from embedder import get_embedding
from xinference.client import Client

COLLECTION_NAME = "knowledge_vectors"
client = Client("http://127.0.0.1:9997")
llm = client.get_model("你的模型ID")  # 替换成你的大模型 ID

def ask_rag(question: str):
    query_vector = get_embedding(question)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=5,
        output_fields=["text"]
    )

    retrieved = [hit.entity.get("text") for hit in results[0]]
    context = "\n".join(retrieved)

    prompt = f"根据以下资料回答问题：\n\n{context}\n\n问题：{question}"
    response = llm.generate(prompt=prompt)

    return {"answer": response["text"]}
