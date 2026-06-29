"""
06_google_gemini.py
===================
Google Generative AI calls for all gemini-* models — alphabetical.

Install:
    pip install google-generativeai

Environment:
    export GOOGLE_API_KEY="AIza..."
"""

import os
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
DUMMY_PROMPT = "What is the capital of France?"

genai.configure(api_key=GOOGLE_API_KEY)


def _call(model_name: str) -> genai.types.GenerateContentResponse:
    model = genai.GenerativeModel(model_name)
    return model.generate_content(DUMMY_PROMPT)


# gemini-1.0-pro
resp = _call("gemini-1.0-pro")

# gemini-1.5-flash
resp = _call("gemini-1.5-flash")

# gemini-1.5-pro
resp = _call("gemini-1.5-pro")

# gemini-2.0-flash
resp = _call("gemini-2.0-flash")

# gemini-2.5-flash
resp = _call("gemini-2.5-flash")

# gemini-2.5-pro
resp = _call("gemini-2.5-pro")

# gemini-3-pro-preview
resp = _call("gemini-3-pro-preview")
