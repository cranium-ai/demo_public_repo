from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key="api_key")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Summarize the following PDF: "}],
)

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "Summarize the following PDF: "}],
)
