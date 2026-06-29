"""
07_openai.py
============
OpenAI API calls for all gpt-* and o<int>-* models — alphabetical.

Install:
    pip install openai

Environment:
    export OPENAI_API_KEY="sk-..."
"""

import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
DUMMY_PROMPT = "What is the capital of France?"

client = OpenAI(api_key=OPENAI_API_KEY)


def _call(model_name: str):
    return client.chat.completions.create(
        model=model_name,
        max_tokens=64,
        messages=[{"role": "user", "content": DUMMY_PROMPT}],
    )


# gpt-3.5-turbo
resp = _call("gpt-3.5-turbo")

# gpt-3.5-turbo-16k
resp = _call("gpt-3.5-turbo-16k")

# gpt-4
resp = _call("gpt-4")

# gpt-4-turbo
resp = _call("gpt-4-turbo")

# gpt-4.1
resp = _call("gpt-4.1")

# gpt-4.1-mini
resp = _call("gpt-4.1-mini")

# gpt-4.1-nano
resp = _call("gpt-4.1-nano")

# gpt-4o
resp = _call("gpt-4o")

# gpt-4o-mini
resp = _call("gpt-4o-mini")

# gpt-5
resp = _call("gpt-5")

# gpt-5-mini
resp = _call("gpt-5-mini")

# gpt-5-nano
resp = _call("gpt-5-nano")

# o1-mini
resp = _call("o1-mini")

# o1-pro
resp = _call("o1-pro")

# o3-mini
resp = _call("o3-mini")

# o4-mini
resp = _call("o4-mini")
