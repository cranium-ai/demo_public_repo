from transformers import pipeline
# ── HuggingFaceM4 ──────────────────────────────────────────────────────────
# HuggingFaceM4/idefics-80b
pipeline("image-to-text", model="HuggingFaceM4/idefics-80b")(DUMMY_IMAGE_PATH)

# HuggingFaceM4/idefics-9b
pipeline("image-to-text", model="HuggingFaceM4/idefics-9b")(DUMMY_IMAGE_PATH)

# HuggingFaceM4/idefics-9b-instruct
pipeline("image-to-text", model="HuggingFaceM4/idefics-9b-instruct")(DUMMY_IMAGE_PATH)

# ── j-hartmann ─────────────────────────────────────────────────────────────
# j-hartmann/emotion-english-distilroberta-base
pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# ── llava-hf ───────────────────────────────────────────────────────────────
# llava-hf/llava-1.5-13b-hf
pipeline("image-to-text", model="llava-hf/llava-1.5-13b-hf")(DUMMY_IMAGE_PATH)

# llava-hf/llava-v1.6-mistral-7b-hf
pipeline("image-to-text", model="llava-hf/llava-v1.6-mistral-7b-hf")(DUMMY_IMAGE_PATH)

# ── lmsys ──────────────────────────────────────────────────────────────────
# lmsys/vicuna-13b-v1.5
pipeline("text-generation", model="lmsys/vicuna-13b-v1.5")

# lmsys/vicuna-7b-v1.5
pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")

# ── meta-llama ─────────────────────────────────────────────────────────────
# meta-llama/Llama-2-13b-chat-hf
pipeline("text-generation", model="meta-llama/Llama-2-13b-chat-hf")

# meta-llama/Llama-2-70b-chat-hf
pipeline("text-generation", model="meta-llama/Llama-2-70b-chat-hf")

# meta-llama/Llama-2-7b-chat-hf
pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

# meta-llama/Llama-3.1-405B-Instruct-FP8
pipeline("text-generation", model="meta-llama/Llama-3.1-405B-Instruct-FP8")

# meta-llama/Llama-3.1-70B
pipeline("text-generation", model="meta-llama/Llama-3.1-70B")

# meta-llama/Llama-3.1-70B-Instruct
pipeline("text-generation", model="meta-llama/Llama-3.1-70B-Instruct")

# meta-llama/Llama-3.2-1B
pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

# meta-llama/Llama-3.2-1B-Instruct
pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

# meta-llama/Llama-3.2-3B
pipeline("text-generation", model="meta-llama/Llama-3.2-3B")

# meta-llama/Llama-3.2-3B-Instruct
pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")

# meta-llama/Llama-3.3-70B-Instruct
pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct")

# meta-llama/Llama-4-Scout-17B-16E-Instruct
pipeline("text-generation", model="meta-llama/Llama-4-Scout-17B-16E-Instruct")

# meta-llama/Meta-Llama-3-70B
pipeline("text-generation", model="meta-llama/Meta-Llama-3-70B")

# meta-llama/Meta-Llama-3-70B-Instruct
pipeline("text-generation", model="meta-llama/Meta-Llama-3-70B-Instruct")

# meta-llama/Meta-Llama-3-8B
pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

# meta-llama/Meta-Llama-3-8B-Instruct
pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")

# meta-llama/Meta-Llama-3.1-8B
pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B")

# meta-llama/Meta-Llama-3.1-8B-Instruct
pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")

# ── microsoft ──────────────────────────────────────────────────────────────
# microsoft/deberta-base
pipeline("fill-mask", model="microsoft/deberta-base")

# microsoft/deberta-v3-base
pipeline("fill-mask", model="microsoft/deberta-v3-base")

# microsoft/DialoGPT-medium
pipeline("text-generation", model="microsoft/DialoGPT-medium")

# microsoft/focalnet-tiny-lrf
pipeline("image-classification", model="microsoft/focalnet-tiny-lrf")(DUMMY_IMAGE_PATH)

# microsoft/Phi-3-medium-4k-instruct
pipeline("text-generation", model="microsoft/Phi-3-medium-4k-instruct")

# microsoft/Phi-3-mini-4k-instruct
pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

# microsoft/phi-4
pipeline("text-generation", model="microsoft/phi-4")

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
pipeline("text-generation", model="mistralai/Codestral-22B-v0.1")

# mistralai/Ministral-8B-Instruct-2410
pipeline("text-generation", model="mistralai/Ministral-8B-Instruct-2410")

# mistralai/Mistral-7B-Instruct-v0.1
pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

# mistralai/Mistral-7B-Instruct-v0.2
pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# mistralai/Mistral-7B-Instruct-v0.3
pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

# mistralai/Mistral-7B-v0.3
pipeline("text-generation", model="mistralai/Mistral-7B-v0.3")

# mistralai/Mistral-Large-Instruct-2407
pipeline("text-generation", model="mistralai/Mistral-Large-Instruct-2407")

# mistralai/Mistral-Small-3.2-24B-Instruct-2506
pipeline("text-generation", model="mistralai/Mistral-Small-3.2-24B-Instruct-2506")

# mistralai/Mixtral-8x22B-Instruct-v0.1
pipeline("text-generation", model="mistralai/Mixtral-8x22B-Instruct-v0.1")

# mistralai/Mixtral-8x7B-Instruct-v0.1
pipeline("text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1")

# mistralai/Mixtral-8x7B-v0.1
pipeline("text-generation", model="mistralai/Mixtral-8x7B-v0.1")