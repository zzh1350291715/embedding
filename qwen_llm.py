import dashscope

dashscope.api_key ='sk-1a7cb72799794d22a00c87e8f76ee7fe'

def call_qwen(prompt: str) -> str:
    response = dashscope.Generation.call(
        model='qwen-max',
        prompt=prompt,
        result_format='text'
    )
    return response.output['text']