# =============================================================================
# 9.  HUGGING FACE  (all model-ids that contain '/')  – alphabetical
# =============================================================================

from transformers import pipeline

# ── 01-ai ──────────────────────────────────────────────────────────────────
# 01-ai/Yi-34B
pipeline("text-generation", model="01-ai/Yi-34B")(DUMMY_PROMPT, max_new_tokens=32)

# 01-ai/Yi-34B-Chat
pipeline("text-generation", model="01-ai/Yi-34B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# ── allenai ────────────────────────────────────────────────────────────────
# allenai/OLMo-7B
pipeline("text-generation", model="allenai/OLMo-7B")(DUMMY_PROMPT, max_new_tokens=32)

# allenai/OLMo-7B-Instruct
pipeline("text-generation", model="allenai/OLMo-7B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# allenai/OLMo-7B-Twin-2T
pipeline("text-generation", model="allenai/OLMo-7B-Twin-2T")(DUMMY_PROMPT, max_new_tokens=32)

# allenai/scibert_scivocab_cased
pipeline("fill-mask", model="allenai/scibert_scivocab_cased")(DUMMY_TEXT)

# allenai/scibert_scivocab_uncased
pipeline("fill-mask", model="allenai/scibert_scivocab_uncased")(DUMMY_TEXT)

# ── arcee-ai ───────────────────────────────────────────────────────────────
# arcee-ai/saul-zephyr-7b-slerp
pipeline("text-generation", model="arcee-ai/saul-zephyr-7b-slerp")(DUMMY_PROMPT, max_new_tokens=32)

# ── bigcode ────────────────────────────────────────────────────────────────
# bigcode/santacoder
pipeline("text-generation", model="bigcode/santacoder")(DUMMY_PROMPT, max_new_tokens=32)

# bigcode/starcoder
pipeline("text-generation", model="bigcode/starcoder")(DUMMY_PROMPT, max_new_tokens=32)

# ── bigscience ─────────────────────────────────────────────────────────────
# bigscience/bloom-1b1
pipeline("text-generation", model="bigscience/bloom-1b1")(DUMMY_PROMPT, max_new_tokens=32)

# bigscience/bloom-1b7
pipeline("text-generation", model="bigscience/bloom-1b7")(DUMMY_PROMPT, max_new_tokens=32)

# bigscience/bloom-3b
pipeline("text-generation", model="bigscience/bloom-3b")(DUMMY_PROMPT, max_new_tokens=32)

# bigscience/bloom-560m
pipeline("text-generation", model="bigscience/bloom-560m")(DUMMY_PROMPT, max_new_tokens=32)

# bigscience/bloom-7b1
pipeline("text-generation", model="bigscience/bloom-7b1")(DUMMY_PROMPT, max_new_tokens=32)

# bigscience/mt0-base
pipeline("text2text-generation", model="bigscience/mt0-base")(DUMMY_PROMPT, max_new_tokens=32)

# bigscience/mt0-small
pipeline("text2text-generation", model="bigscience/mt0-small")(DUMMY_PROMPT, max_new_tokens=32)

# ── cardiffnlp ─────────────────────────────────────────────────────────────
# cardiffnlp/twitter-roberta-base-sentiment-latest
pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")(DUMMY_TEXT)

# ── codellama ──────────────────────────────────────────────────────────────
# codellama/CodeLlama-13b-hf
pipeline("text-generation", model="codellama/CodeLlama-13b-hf")(DUMMY_PROMPT, max_new_tokens=32)

# codellama/CodeLlama-13b-Instruct-hf
pipeline("text-generation", model="codellama/CodeLlama-13b-Instruct-hf")(DUMMY_PROMPT, max_new_tokens=32)

# codellama/CodeLlama-34b-hf
pipeline("text-generation", model="codellama/CodeLlama-34b-hf")(DUMMY_PROMPT, max_new_tokens=32)

# codellama/CodeLlama-34b-Instruct-hf
pipeline("text-generation", model="codellama/CodeLlama-34b-Instruct-hf")(DUMMY_PROMPT, max_new_tokens=32)

# codellama/CodeLlama-70b-hf
pipeline("text-generation", model="codellama/CodeLlama-70b-hf")(DUMMY_PROMPT, max_new_tokens=32)

# codellama/CodeLlama-70b-Instruct-hf
pipeline("text-generation", model="codellama/CodeLlama-70b-Instruct-hf")(DUMMY_PROMPT, max_new_tokens=32)

# codellama/CodeLlama-7b-hf
pipeline("text-generation", model="codellama/CodeLlama-7b-hf")(DUMMY_PROMPT, max_new_tokens=32)

# codellama/CodeLlama-7b-Instruct-hf
pipeline("text-generation", model="codellama/CodeLlama-7b-Instruct-hf")(DUMMY_PROMPT, max_new_tokens=32)

# ── CohereForAI ────────────────────────────────────────────────────────────
# CohereForAI/c4ai-command-r7b-12-2024
pipeline("text-generation", model="CohereForAI/c4ai-command-r7b-12-2024")(DUMMY_PROMPT, max_new_tokens=32)

# ── deepseek-ai ────────────────────────────────────────────────────────────
# deepseek-ai/deepseek-coder-33b-instruct
pipeline("text-generation", model="deepseek-ai/deepseek-coder-33b-instruct")(DUMMY_PROMPT, max_new_tokens=32)

# deepseek-ai/deepseek-llm-67b-chat
pipeline("text-generation", model="deepseek-ai/deepseek-llm-67b-chat")(DUMMY_PROMPT, max_new_tokens=32)

# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")(DUMMY_PROMPT, max_new_tokens=32)

# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")(DUMMY_PROMPT, max_new_tokens=32)

# ── distilbert ─────────────────────────────────────────────────────────────
# distilbert/distilbert-base-uncased
pipeline("fill-mask", model="distilbert/distilbert-base-uncased")(DUMMY_TEXT)

# distilbert/distilroberta-base
pipeline("fill-mask", model="distilbert/distilroberta-base")(DUMMY_TEXT)

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
pipeline("fill-mask", model="FacebookAI/roberta-base")(DUMMY_TEXT)

# FacebookAI/roberta-large
pipeline("fill-mask", model="FacebookAI/roberta-large")(DUMMY_TEXT)

# ── google ─────────────────────────────────────────────────────────────────
# google-bert/bert-base-cased
pipeline("fill-mask", model="google-bert/bert-base-cased")(DUMMY_TEXT)

# google-bert/bert-base-uncased
pipeline("fill-mask", model="google-bert/bert-base-uncased")(DUMMY_TEXT)

# google-bert/bert-large-uncased
pipeline("fill-mask", model="google-bert/bert-large-uncased")(DUMMY_TEXT)

# google-t5/t5-base
pipeline("text2text-generation", model="google-t5/t5-base")(DUMMY_PROMPT, max_new_tokens=32)

# google-t5/t5-large
pipeline("text2text-generation", model="google-t5/t5-large")(DUMMY_PROMPT, max_new_tokens=32)

# google-t5/t5-small
pipeline("text2text-generation", model="google-t5/t5-small")(DUMMY_PROMPT, max_new_tokens=32)

# google/efficientnet-b0
pipeline("image-classification", model="google/efficientnet-b0")(DUMMY_IMAGE_PATH)

# google/efficientnet-b7
pipeline("image-classification", model="google/efficientnet-b7")(DUMMY_IMAGE_PATH)

# google/flan-t5-xxl
pipeline("text2text-generation", model="google/flan-t5-xxl")(DUMMY_PROMPT, max_new_tokens=32)

# google/gemma-2-9b
pipeline("text-generation", model="google/gemma-2-9b")(DUMMY_PROMPT, max_new_tokens=32)

# google/gemma-2-9b-it
pipeline("text-generation", model="google/gemma-2-9b-it")(DUMMY_PROMPT, max_new_tokens=32)

# google/gemma-2b-it
pipeline("text-generation", model="google/gemma-2b-it")(DUMMY_PROMPT, max_new_tokens=32)

# google/gemma-3-12b-it
pipeline("text-generation", model="google/gemma-3-12b-it")(DUMMY_PROMPT, max_new_tokens=32)

# google/gemma-3-27b-it
pipeline("text-generation", model="google/gemma-3-27b-it")(DUMMY_PROMPT, max_new_tokens=32)

# google/gemma-7b-it
pipeline("text-generation", model="google/gemma-7b-it")(DUMMY_PROMPT, max_new_tokens=32)

# google/pegasus-large
pipeline("summarization", model="google/pegasus-large")("Hugging Face is a technology company.", max_new_tokens=32)

# ── HuggingFaceM4 ──────────────────────────────────────────────────────────
# HuggingFaceM4/idefics-80b
pipeline("image-to-text", model="HuggingFaceM4/idefics-80b")(DUMMY_IMAGE_PATH)

# HuggingFaceM4/idefics-9b
pipeline("image-to-text", model="HuggingFaceM4/idefics-9b")(DUMMY_IMAGE_PATH)

# HuggingFaceM4/idefics-9b-instruct
pipeline("image-to-text", model="HuggingFaceM4/idefics-9b-instruct")(DUMMY_IMAGE_PATH)

# ── j-hartmann ─────────────────────────────────────────────────────────────
# j-hartmann/emotion-english-distilroberta-base
pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")(DUMMY_TEXT)

# ── llava-hf ───────────────────────────────────────────────────────────────
# llava-hf/llava-1.5-13b-hf
pipeline("image-to-text", model="llava-hf/llava-1.5-13b-hf")(DUMMY_IMAGE_PATH)

# llava-hf/llava-v1.6-mistral-7b-hf
pipeline("image-to-text", model="llava-hf/llava-v1.6-mistral-7b-hf")(DUMMY_IMAGE_PATH)

# ── lmsys ──────────────────────────────────────────────────────────────────
# lmsys/vicuna-13b-v1.5
pipeline("text-generation", model="lmsys/vicuna-13b-v1.5")(DUMMY_PROMPT, max_new_tokens=32)

# lmsys/vicuna-7b-v1.5
pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")(DUMMY_PROMPT, max_new_tokens=32)

# ── meta-llama ─────────────────────────────────────────────────────────────
# meta-llama/Llama-2-13b-chat-hf
pipeline("text-generation", model="meta-llama/Llama-2-13b-chat-hf")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-2-70b-chat-hf
pipeline("text-generation", model="meta-llama/Llama-2-70b-chat-hf")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-2-7b-chat-hf
pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-3.1-405B-Instruct-FP8
pipeline("text-generation", model="meta-llama/Llama-3.1-405B-Instruct-FP8")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-3.1-70B
pipeline("text-generation", model="meta-llama/Llama-3.1-70B")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-3.1-70B-Instruct
pipeline("text-generation", model="meta-llama/Llama-3.1-70B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-3.2-1B
pipeline("text-generation", model="meta-llama/Llama-3.2-1B")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-3.2-1B-Instruct
pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-3.2-3B
pipeline("text-generation", model="meta-llama/Llama-3.2-3B")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-3.2-3B-Instruct
pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-3.3-70B-Instruct
pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Llama-4-Scout-17B-16E-Instruct
pipeline("text-generation", model="meta-llama/Llama-4-Scout-17B-16E-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Meta-Llama-3-70B
pipeline("text-generation", model="meta-llama/Meta-Llama-3-70B")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Meta-Llama-3-70B-Instruct
pipeline("text-generation", model="meta-llama/Meta-Llama-3-70B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Meta-Llama-3-8B
pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Meta-Llama-3-8B-Instruct
pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Meta-Llama-3.1-8B
pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B")(DUMMY_PROMPT, max_new_tokens=32)

# meta-llama/Meta-Llama-3.1-8B-Instruct
pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# ── microsoft ──────────────────────────────────────────────────────────────
# microsoft/deberta-base
pipeline("fill-mask", model="microsoft/deberta-base")(DUMMY_TEXT)

# microsoft/deberta-v3-base
pipeline("fill-mask", model="microsoft/deberta-v3-base")(DUMMY_TEXT)

# microsoft/DialoGPT-medium
pipeline("text-generation", model="microsoft/DialoGPT-medium")(DUMMY_PROMPT, max_new_tokens=32)

# microsoft/focalnet-tiny-lrf
pipeline("image-classification", model="microsoft/focalnet-tiny-lrf")(DUMMY_IMAGE_PATH)

# microsoft/Phi-3-medium-4k-instruct
pipeline("text-generation", model="microsoft/Phi-3-medium-4k-instruct")(DUMMY_PROMPT, max_new_tokens=32)

# microsoft/Phi-3-mini-4k-instruct
pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")(DUMMY_PROMPT, max_new_tokens=32)

# microsoft/phi-4
pipeline("text-generation", model="microsoft/phi-4")(DUMMY_PROMPT, max_new_tokens=32)

# microsoft/resnet-101
pipeline("image-classification", model="microsoft/resnet-101")(DUMMY_IMAGE_PATH)

# microsoft/resnet-152
pipeline("image-classification", model="microsoft/resnet-152")(DUMMY_IMAGE_PATH)

# microsoft/resnet-18
pipeline("image-classification", model="microsoft/resnet-18")(DUMMY_IMAGE_PATH)

# microsoft/resnet-50
pipeline("image-classification", model="microsoft/resnet-50")(DUMMY_IMAGE_PATH)

# ── mistralai ──────────────────────────────────────────────────────────────
# mistralai/Codestral-22B-v0.1
pipeline("text-generation", model="mistralai/Codestral-22B-v0.1")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Ministral-8B-Instruct-2410
pipeline("text-generation", model="mistralai/Ministral-8B-Instruct-2410")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mistral-7B-Instruct-v0.1
pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mistral-7B-Instruct-v0.2
pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mistral-7B-Instruct-v0.3
pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mistral-7B-v0.3
pipeline("text-generation", model="mistralai/Mistral-7B-v0.3")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mistral-Large-Instruct-2407
pipeline("text-generation", model="mistralai/Mistral-Large-Instruct-2407")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mistral-Small-3.2-24B-Instruct-2506
pipeline("text-generation", model="mistralai/Mistral-Small-3.2-24B-Instruct-2506")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mixtral-8x22B-Instruct-v0.1
pipeline("text-generation", model="mistralai/Mixtral-8x22B-Instruct-v0.1")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mixtral-8x7B-Instruct-v0.1
pipeline("text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1")(DUMMY_PROMPT, max_new_tokens=32)

# mistralai/Mixtral-8x7B-v0.1
pipeline("text-generation", model="mistralai/Mixtral-8x7B-v0.1")(DUMMY_PROMPT, max_new_tokens=32)

# ── NousResearch ───────────────────────────────────────────────────────────
# NousResearch/Nous-Capybara-7B-V1.9
pipeline("text-generation", model="NousResearch/Nous-Capybara-7B-V1.9")(DUMMY_PROMPT, max_new_tokens=32)

# NousResearch/Nous-Hermes-llama-2-7b
pipeline("text-generation", model="NousResearch/Nous-Hermes-llama-2-7b")(DUMMY_PROMPT, max_new_tokens=32)

# NousResearch/Nous-Hermes-Llama2-13b
pipeline("text-generation", model="NousResearch/Nous-Hermes-Llama2-13b")(DUMMY_PROMPT, max_new_tokens=32)

# ── nreimers ───────────────────────────────────────────────────────────────
# nreimers/BERT-Tiny_L-2_H-128_A-2
pipeline("fill-mask", model="nreimers/BERT-Tiny_L-2_H-128_A-2")(DUMMY_TEXT)

# nreimers/MiniLM-L6-H384-uncased
pipeline("feature-extraction", model="nreimers/MiniLM-L6-H384-uncased")(DUMMY_TEXT)

# ── NumbersStation ─────────────────────────────────────────────────────────
# NumbersStation/nsql-llama-2-7B
pipeline("text-generation", model="NumbersStation/nsql-llama-2-7B")(DUMMY_PROMPT, max_new_tokens=32)

# ── nvidia ─────────────────────────────────────────────────────────────────
# nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
pipeline("text-generation", model="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5")(DUMMY_PROMPT, max_new_tokens=32)

# nvidia/parakeet-tdt-0.6b-v3  (ASR model)
pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")
# pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")("audio.wav")

# ── Open-Orca ──────────────────────────────────────────────────────────────
# Open-Orca/Mistral-7B-OpenOrca
pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca")(DUMMY_PROMPT, max_new_tokens=32)

# ── openai (HF namespace) ──────────────────────────────────────────────────
# openai/gpt-oss-120b
pipeline("text-generation", model="openai/gpt-oss-120b")(DUMMY_PROMPT, max_new_tokens=32)

# openai/gpt-oss-20b
pipeline("text-generation", model="openai/gpt-oss-20b")(DUMMY_PROMPT, max_new_tokens=32)

# openai/whisper-base  (ASR)
pipeline("automatic-speech-recognition", model="openai/whisper-base")
# pipeline("automatic-speech-recognition", model="openai/whisper-base")("audio.wav")

# openai/whisper-large-v3-turbo
pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
# pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")("audio.wav")

# ── openchat ───────────────────────────────────────────────────────────────
# openchat/openchat-3.5-1210
pipeline("text-generation", model="openchat/openchat-3.5-1210")(DUMMY_PROMPT, max_new_tokens=32)

# ── prajjwal1 ──────────────────────────────────────────────────────────────
# prajjwal1/bert-tiny
pipeline("fill-mask", model="prajjwal1/bert-tiny")(DUMMY_TEXT)

# ── Qwen ───────────────────────────────────────────────────────────────────
# Qwen/Qwen1.5-0.5B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-0.5B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen1.5-1.8B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-1.8B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen1.5-110B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-110B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen1.5-14B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-14B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen1.5-32B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-32B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen1.5-4B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-4B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen1.5-72B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-72B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen1.5-7B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-7B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen2-72B-Instruct
pipeline("text-generation", model="Qwen/Qwen2-72B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen2-7B-Instruct
pipeline("text-generation", model="Qwen/Qwen2-7B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen2.5-32B
pipeline("text-generation", model="Qwen/Qwen2.5-32B")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen3-4B
pipeline("text-generation", model="Qwen/Qwen3-4B")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/Qwen3-Next-80B-A3B-Instruct
pipeline("text-generation", model="Qwen/Qwen3-Next-80B-A3B-Instruct")(DUMMY_PROMPT, max_new_tokens=32)

# Qwen/QwQ-32B
pipeline("text-generation", model="Qwen/QwQ-32B")(DUMMY_PROMPT, max_new_tokens=32)

# ── stabilityai ────────────────────────────────────────────────────────────
# stabilityai/stablelm-tuned-alpha-3b
pipeline("text-generation", model="stabilityai/stablelm-tuned-alpha-3b")(DUMMY_PROMPT, max_new_tokens=32)

# ── tiiuae ─────────────────────────────────────────────────────────────────
# tiiuae/falcon-7b-instruct
pipeline("text-generation", model="tiiuae/falcon-7b-instruct")(DUMMY_PROMPT, max_new_tokens=32)

# ── timm (namespace/repo models) ───────────────────────────────────────────
# timm/eva_giant_patch14_224.clip_ft_in1k
pipeline("image-classification", model="timm/eva_giant_patch14_224.clip_ft_in1k")(DUMMY_IMAGE_PATH)

# timm/inception_v4.tf_in1k
pipeline("image-classification", model="timm/inception_v4.tf_in1k")(DUMMY_IMAGE_PATH)

# timm/levit_128s.fb_dist_in1k
pipeline("image-classification", model="timm/levit_128s.fb_dist_in1k")(DUMMY_IMAGE_PATH)

# timm/tf_efficientnet_lite0.in1k
pipeline("image-classification", model="timm/tf_efficientnet_lite0.in1k")(DUMMY_IMAGE_PATH)

# ── TinyLlama ──────────────────────────────────────────────────────────────
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")(DUMMY_PROMPT, max_new_tokens=32)

# ── togethercomputer ───────────────────────────────────────────────────────
# togethercomputer/RedPajama-INCITE-7B-Chat
pipeline("text-generation", model="togethercomputer/RedPajama-INCITE-7B-Chat")(DUMMY_PROMPT, max_new_tokens=32)

# togethercomputer/RedPajama-INCITE-Chat-3B-v1
pipeline("text-generation", model="togethercomputer/RedPajama-INCITE-Chat-3B-v1")(DUMMY_PROMPT, max_new_tokens=32)

# ── unsloth ────────────────────────────────────────────────────────────────
# unsloth/DeepSeek-R1-GGUF
pipeline("text-generation", model="unsloth/DeepSeek-R1-GGUF")(DUMMY_PROMPT, max_new_tokens=32)
