import time

from openai import OpenAI


client = OpenAI(base_url="http://127.0.0.1:10000/v1", api_key="not-needed")
messages = []

print("AI Investment Orchestrator")
print("Ask about one company's news, financials, or both. Type exit to stop.")

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if user_input.lower() in {"exit", "quit"}:
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    # start time
    start_time = time.time()

    response = client.chat.completions.create(
        model="vertical-api-orchestrator",
        messages=messages,
        user="api-world-demo",
    )

    # end time
    end_time = time.time()
    print(f"Response time: {end_time - start_time:.2f} seconds")

    assistant_text = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": assistant_text})
    print(f"\nOrchestrator:\n{assistant_text}")
