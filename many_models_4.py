# =============================================================================
# 9.  HUGGING FACE  (all model-ids that contain '/')  – alphabetical
# =============================================================================

from transformers import pipeline

# ── 01-ai ──────────────────────────────────────────────────────────────────
# 01-ai/Yi-34B
pipeline("text-generation", model="01-ai/Yi-34B")

# 01-ai/Yi-34B-Chat
pipeline("text-generation", model="01-ai/Yi-34B-Chat")

# ── allenai ────────────────────────────────────────────────────────────────
# allenai/OLMo-7B
pipeline("text-generation", model="allenai/OLMo-7B")

# allenai/OLMo-7B-Instruct
pipeline("text-generation", model="allenai/OLMo-7B-Instruct")

# allenai/OLMo-7B-Twin-2T
pipeline("text-generation", model="allenai/OLMo-7B-Twin-2T")

# allenai/scibert_scivocab_cased
pipeline("fill-mask", model="allenai/scibert_scivocab_cased")

# allenai/scibert_scivocab_uncased
pipeline("fill-mask", model="allenai/scibert_scivocab_uncased")

# ── arcee-ai ───────────────────────────────────────────────────────────────
# arcee-ai/saul-zephyr-7b-slerp
pipeline("text-generation", model="arcee-ai/saul-zephyr-7b-slerp")

# ── bigcode ────────────────────────────────────────────────────────────────
# bigcode/santacoder
pipeline("text-generation", model="bigcode/santacoder")

# bigcode/starcoder
pipeline("text-generation", model="bigcode/starcoder")

# ── bigscience ─────────────────────────────────────────────────────────────
# bigscience/bloom-1b1
pipeline("text-generation", model="bigscience/bloom-1b1")

# bigscience/bloom-1b7
pipeline("text-generation", model="bigscience/bloom-1b7")

# bigscience/bloom-3b
pipeline("text-generation", model="bigscience/bloom-3b")

# bigscience/bloom-560m
pipeline("text-generation", model="bigscience/bloom-560m")

# bigscience/bloom-7b1
pipeline("text-generation", model="bigscience/bloom-7b1")

# bigscience/mt0-base
pipeline("text2text-generation", model="bigscience/mt0-base")

# bigscience/mt0-small
pipeline("text2text-generation", model="bigscience/mt0-small")

# ── cardiffnlp ─────────────────────────────────────────────────────────────
# cardiffnlp/twitter-roberta-base-sentiment-latest
pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# ── codellama ──────────────────────────────────────────────────────────────
# codellama/CodeLlama-13b-hf
pipeline("text-generation", model="codellama/CodeLlama-13b-hf")

# codellama/CodeLlama-13b-Instruct-hf
pipeline("text-generation", model="codellama/CodeLlama-13b-Instruct-hf")

# codellama/CodeLlama-34b-hf
pipeline("text-generation", model="codellama/CodeLlama-34b-hf")

# codellama/CodeLlama-34b-Instruct-hf
pipeline("text-generation", model="codellama/CodeLlama-34b-Instruct-hf")

# codellama/CodeLlama-70b-hf
pipeline("text-generation", model="codellama/CodeLlama-70b-hf")

# codellama/CodeLlama-70b-Instruct-hf
pipeline("text-generation", model="codellama/CodeLlama-70b-Instruct-hf")

# codellama/CodeLlama-7b-hf
pipeline("text-generation", model="codellama/CodeLlama-7b-hf")

# codellama/CodeLlama-7b-Instruct-hf
pipeline("text-generation", model="codellama/CodeLlama-7b-Instruct-hf")

# ── CohereForAI ────────────────────────────────────────────────────────────
# CohereForAI/c4ai-command-r7b-12-2024
pipeline("text-generation", model="CohereForAI/c4ai-command-r7b-12-2024")

# ── deepseek-ai ────────────────────────────────────────────────────────────
# deepseek-ai/deepseek-coder-33b-instruct
pipeline("text-generation", model="deepseek-ai/deepseek-coder-33b-instruct")

# deepseek-ai/deepseek-llm-67b-chat
pipeline("text-generation", model="deepseek-ai/deepseek-llm-67b-chat")

# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

# ── distilbert ─────────────────────────────────────────────────────────────
# distilbert/distilbert-base-uncased
pipeline("fill-mask", model="distilbert/distilbert-base-uncased")

# distilbert/distilroberta-base
pipeline("fill-mask", model="distilbert/distilroberta-base")

# ── facebook ───────────────────────────────────────────────────────────────
# facebook/bart-large
pipeline("summarization", model="facebook/bart-large")("Hugging Face is a technology company.", max_new_tokens=32)

# facebook/blenderbot-400M-distill
pipeline("conversational", model="facebook/blenderbot-400M-distill")(DUMMY_PROMPT)

# facebook/regnet-x-016
pipeline("image-classification", model="facebook/regnet-x-016")(DUMMY_IMAGE_PATH)

# facebook/regnet-x-032
pipeline("image-classification", model="facebook/regnet-x-032")(DUMMY_IMAGE_PATH)

# facebook/regnet-x-040
pipeline("image-classification", model="facebook/regnet-x-040")(DUMMY_IMAGE_PATH)

# facebook/regnet-x-064
pipeline("image-classification", model="facebook/regnet-x-064")(DUMMY_IMAGE_PATH)

# facebook/regnet-x-080
pipeline("image-classification", model="facebook/regnet-x-080")(DUMMY_IMAGE_PATH)

# facebook/regnet-x-120
pipeline("image-classification", model="facebook/regnet-x-120")(DUMMY_IMAGE_PATH)

# ── FacebookAI ─────────────────────────────────────────────────────────────
# FacebookAI/roberta-base
pipeline("fill-mask", model="FacebookAI/roberta-base")

# FacebookAI/roberta-large
pipeline("fill-mask", model="FacebookAI/roberta-large")

# ── google ─────────────────────────────────────────────────────────────────
# google-bert/bert-base-cased
pipeline("fill-mask", model="google-bert/bert-base-cased")

# google-bert/bert-base-uncased
pipeline("fill-mask", model="google-bert/bert-base-uncased")

# google-bert/bert-large-uncased
pipeline("fill-mask", model="google-bert/bert-large-uncased")

# google-t5/t5-base
pipeline("text2text-generation", model="google-t5/t5-base")

# google-t5/t5-large
pipeline("text2text-generation", model="google-t5/t5-large")

# google-t5/t5-small
pipeline("text2text-generation", model="google-t5/t5-small")

# google/efficientnet-b0
pipeline("image-classification", model="google/efficientnet-b0")(DUMMY_IMAGE_PATH)

# google/efficientnet-b7
pipeline("image-classification", model="google/efficientnet-b7")(DUMMY_IMAGE_PATH)

# google/flan-t5-xxl
pipeline("text2text-generation", model="google/flan-t5-xxl")

# google/gemma-2-9b
pipeline("text-generation", model="google/gemma-2-9b")

# google/gemma-2-9b-it
pipeline("text-generation", model="google/gemma-2-9b-it")

# google/gemma-2b-it
pipeline("text-generation", model="google/gemma-2b-it")

# google/gemma-3-12b-it
pipeline("text-generation", model="google/gemma-3-12b-it")

# google/gemma-3-27b-it
pipeline("text-generation", model="google/gemma-3-27b-it")

# google/gemma-7b-it
pipeline("text-generation", model="google/gemma-7b-it")

# google/pegasus-large
pipeline("summarization", model="google/pegasus-large")("Hugging Face is a technology company.", max_new_tokens=32)
