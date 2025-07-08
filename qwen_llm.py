from dashscope import Generation, MultiModalConversation
import dashscope
from dashscope import Generation

dashscope.api_key = 'sk-1a7cb72799794d22a00c87e8f76ee7fe'

# 文本模型调用（如 qwen-max）
def call_qwen(prompt: str) -> str:
    response = Generation.call(
        model='qwen-max',
        prompt=prompt,
        result_format='text'
    )
    return response.output['text']

# 图像问答模型调用（如 qwen-vl-plus）
def call_qwen_vl(image_base64: str, question: str) -> str:
    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=[
            {"role": "user", "content": [
                {"image": {"url": image_base64}},
                {"text": question}
            ]}
        ]
    )
    return response['output']['choices'][0]['message']['content']
