from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="not-needed")
messages = []

prompt = "What is the capital of Belgium?"

messages.append({"role": "user", "content": prompt})
response = client.chat.completions.create(
    model="local-llm",
    messages=messages,
    user="api-world-demo",
)
assistant_text = response.choices[0].message.content or ""
messages.append({"role": "assistant", "content": assistant_text})
print(f"\nSLM Response:\n{assistant_text}")
