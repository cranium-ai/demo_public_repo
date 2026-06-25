# =============================================================================
# 6.  ANTHROPIC (claude-*) MODELS  – alphabetical
# =============================================================================

import anthropic

_anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def _claude_call(model_name: str) -> anthropic.types.Message:
    return _anthropic_client.messages.create(
        model=model_name,
        max_tokens=64,
        messages=[{"role": "user", "content": DUMMY_PROMPT}],
    )

# claude-3-5-sonnet-20240620
resp = _claude_call("claude-3-5-sonnet-20240620")

# claude-3-5-sonnet-20241022
resp = _claude_call("claude-3-5-sonnet-20241022")

# claude-3-7-sonnet-20250219
resp = _claude_call("claude-3-7-sonnet-20250219")

# claude-3-haiku-20240307
resp = _claude_call("claude-3-haiku-20240307")

# claude-haiku-4-5-20251001
resp = _claude_call("claude-haiku-4-5-20251001")

# claude-opus-4-6
resp = _claude_call("claude-opus-4-6")

# claude-sonnet-4-20250514
resp = _claude_call("claude-sonnet-4-20250514")

# claude-sonnet-4-5-20250929
resp = _claude_call("claude-sonnet-4-5-20250929")

# claude-sonnet-4-6
resp = _claude_call("claude-sonnet-4-6")


# =============================================================================
# 7.  GOOGLE GENERATIVE AI (gemini-*) MODELS  – alphabetical
# =============================================================================

import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)

def _gemini_call(model_name: str) -> genai.types.GenerateContentResponse:
    model = genai.GenerativeModel(model_name)
    return model.generate_content(DUMMY_PROMPT)

# gemini-1.0-pro
resp = _gemini_call("gemini-1.0-pro")

# gemini-1.5-flash
resp = _gemini_call("gemini-1.5-flash")

# gemini-1.5-pro
resp = _gemini_call("gemini-1.5-pro")

# gemini-2.0-flash
resp = _gemini_call("gemini-2.0-flash")

# gemini-2.5-flash
resp = _gemini_call("gemini-2.5-flash")

# gemini-2.5-pro
resp = _gemini_call("gemini-2.5-pro")

# gemini-3-pro-preview
resp = _gemini_call("gemini-3-pro-preview")


# =============================================================================
# 8.  OPENAI (gpt-* and o<int>-*) MODELS  – alphabetical
# =============================================================================

from openai import OpenAI

_openai_client = OpenAI(api_key=OPENAI_API_KEY)

def _openai_chat(model_name: str) -> openai.types.chat.ChatCompletion:
    return _openai_client.chat.completions.create(
        model=model_name,
        max_tokens=64,
        messages=[{"role": "user", "content": DUMMY_PROMPT}],
    )

# gpt-3.5-turbo
resp = _openai_chat("gpt-3.5-turbo")

# gpt-3.5-turbo-16k
resp = _openai_chat("gpt-3.5-turbo-16k")

# gpt-4
resp = _openai_chat("gpt-4")

# gpt-4-turbo
resp = _openai_chat("gpt-4-turbo")

# gpt-4.1
resp = _openai_chat("gpt-4.1")

# gpt-4.1-mini
resp = _openai_chat("gpt-4.1-mini")

# gpt-4.1-nano
resp = _openai_chat("gpt-4.1-nano")

# gpt-4o
resp = _openai_chat("gpt-4o")

# gpt-4o-mini
resp = _openai_chat("gpt-4o-mini")

# gpt-5
resp = _openai_chat("gpt-5")

# gpt-5-mini
resp = _openai_chat("gpt-5-mini")

# gpt-5-nano
resp = _openai_chat("gpt-5-nano")

# o1-mini
resp = _openai_chat("o1-mini")

# o1-pro
resp = _openai_chat("o1-pro")

# o3-mini
resp = _openai_chat("o3-mini")

# o4-mini
resp = _openai_chat("o4-mini")