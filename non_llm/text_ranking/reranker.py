import requests
import os
from transformers import AutoModelForSequenceClassification

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


# Model via Nvidia Reranking API
def call_reranking_model(query, passages):
    invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }

    payload = {
        "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        "query": {"text": query},
        "passages": [{"text": passage} for passage in passages],
    }

    # re-use connections
    session = requests.Session()

    response = session.post(invoke_url, headers=headers, json=payload)

    response.raise_for_status()
    response_body = response.json()

    ranked_descending_indices = [
        rank_metadata["index"] for rank_metadata in response_body["rankings"]
    ]
    return ranked_descending_indices


# Model via Hugging Face Transformers (for local inference)
model = AutoModelForSequenceClassification.from_pretrained(
    "nvidia/llama-nemotron-rerank-1b-v2",
).eval()
