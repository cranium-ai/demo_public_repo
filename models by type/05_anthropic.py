"""
05_anthropic.py
===============
Anthropic API calls for all claude-* models — alphabetical.

Install:
    pip install anthropic

Environment:
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os
import anthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY")
DUMMY_PROMPT = "What is the capital of France?"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _call(model_name: str) -> anthropic.types.Message:
    return client.messages.create(
        model=model_name,
        max_tokens=64,
        messages=[{"role": "user", "content": DUMMY_PROMPT}],
    )


# claude-3-5-sonnet-20240620
resp = _call("claude-3-5-sonnet-20240620")

# claude-3-5-sonnet-20241022
resp = _call("claude-3-5-sonnet-20241022")

# claude-3-7-sonnet-20250219
resp = _call("claude-3-7-sonnet-20250219")

# claude-3-haiku-20240307
resp = _call("claude-3-haiku-20240307")

# claude-haiku-4-5-20251001
resp = _call("claude-haiku-4-5-20251001")

# claude-opus-4-6
resp = _call("claude-opus-4-6")

# claude-sonnet-4-20250514
resp = _call("claude-sonnet-4-20250514")

# claude-sonnet-4-5-20250929
resp = _call("claude-sonnet-4-5-20250929")

# claude-sonnet-4-6
resp = _call("claude-sonnet-4-6")
