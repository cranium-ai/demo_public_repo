# ── NousResearch ───────────────────────────────────────────────────────────
# NousResearch/Nous-Capybara-7B-V1.9
pipeline("text-generation", model="NousResearch/Nous-Capybara-7B-V1.9")

# NousResearch/Nous-Hermes-llama-2-7b
pipeline("text-generation", model="NousResearch/Nous-Hermes-llama-2-7b")

# NousResearch/Nous-Hermes-Llama2-13b
pipeline("text-generation", model="NousResearch/Nous-Hermes-Llama2-13b")

# ── nreimers ───────────────────────────────────────────────────────────────
# nreimers/BERT-Tiny_L-2_H-128_A-2
pipeline("fill-mask", model="nreimers/BERT-Tiny_L-2_H-128_A-2")

# nreimers/MiniLM-L6-H384-uncased
pipeline("feature-extraction", model="nreimers/MiniLM-L6-H384-uncased")

# ── NumbersStation ─────────────────────────────────────────────────────────
# NumbersStation/nsql-llama-2-7B
pipeline("text-generation", model="NumbersStation/nsql-llama-2-7B")

# ── nvidia ─────────────────────────────────────────────────────────────────
# nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
pipeline("text-generation", model="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5")

# nvidia/parakeet-tdt-0.6b-v3  (ASR model)
pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")
# pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")("audio.wav")

# ── Open-Orca ──────────────────────────────────────────────────────────────
# Open-Orca/Mistral-7B-OpenOrca
pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca")

# ── openai (HF namespace) ──────────────────────────────────────────────────
# openai/gpt-oss-120b
pipeline("text-generation", model="openai/gpt-oss-120b")

# openai/gpt-oss-20b
pipeline("text-generation", model="openai/gpt-oss-20b")

# openai/whisper-base  (ASR)
pipeline("automatic-speech-recognition", model="openai/whisper-base")
# pipeline("automatic-speech-recognition", model="openai/whisper-base")("audio.wav")

# openai/whisper-large-v3-turbo
pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
# pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")("audio.wav")

# ── openchat ───────────────────────────────────────────────────────────────
# openchat/openchat-3.5-1210
pipeline("text-generation", model="openchat/openchat-3.5-1210")

# ── prajjwal1 ──────────────────────────────────────────────────────────────
# prajjwal1/bert-tiny
pipeline("fill-mask", model="prajjwal1/bert-tiny")

# ── Qwen ───────────────────────────────────────────────────────────────────
# Qwen/Qwen1.5-0.5B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-0.5B-Chat")

# Qwen/Qwen1.5-1.8B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-1.8B-Chat")

# Qwen/Qwen1.5-110B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-110B-Chat")

# Qwen/Qwen1.5-14B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-14B-Chat")

# Qwen/Qwen1.5-32B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-32B-Chat")

# Qwen/Qwen1.5-4B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-4B-Chat")

# Qwen/Qwen1.5-72B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-72B-Chat")

# Qwen/Qwen1.5-7B-Chat
pipeline("text-generation", model="Qwen/Qwen1.5-7B-Chat")

# Qwen/Qwen2-72B-Instruct
pipeline("text-generation", model="Qwen/Qwen2-72B-Instruct")

# Qwen/Qwen2-7B-Instruct
pipeline("text-generation", model="Qwen/Qwen2-7B-Instruct")

# Qwen/Qwen2.5-32B
pipeline("text-generation", model="Qwen/Qwen2.5-32B")

# Qwen/Qwen3-4B
pipeline("text-generation", model="Qwen/Qwen3-4B")

# Qwen/Qwen3-Next-80B-A3B-Instruct
pipeline("text-generation", model="Qwen/Qwen3-Next-80B-A3B-Instruct")

# Qwen/QwQ-32B
pipeline("text-generation", model="Qwen/QwQ-32B")

# ── stabilityai ────────────────────────────────────────────────────────────
# stabilityai/stablelm-tuned-alpha-3b
pipeline("text-generation", model="stabilityai/stablelm-tuned-alpha-3b")

# ── tiiuae ─────────────────────────────────────────────────────────────────
# tiiuae/falcon-7b-instruct
pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

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
pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ── togethercomputer ───────────────────────────────────────────────────────
# togethercomputer/RedPajama-INCITE-7B-Chat
pipeline("text-generation", model="togethercomputer/RedPajama-INCITE-7B-Chat")

# togethercomputer/RedPajama-INCITE-Chat-3B-v1
pipeline("text-generation", model="togethercomputer/RedPajama-INCITE-Chat-3B-v1")

# ── unsloth ────────────────────────────────────────────────────────────────
# unsloth/DeepSeek-R1-GGUF
pipeline("text-generation", model="unsloth/DeepSeek-R1-GGUF")