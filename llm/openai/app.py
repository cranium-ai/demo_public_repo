from openai import OpenAI


client = OpenAI(api_key="YOUR_API_KEY")

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the history of the United States."},
    ],
)

completion2 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the current temperature in Los Angeles?"},
    ],
)
