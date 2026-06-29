"""
08_hf_nlp.py
============
Hugging Face text / NLP models (contain '/' in name) — alphabetical.
Covers: text-generation, fill-mask, text2text-generation, summarization,
        conversational, sentiment-analysis, feature-extraction.

Install:
    pip install transformers torch accelerate sentencepiece
"""

from transformers import pipeline, Conversation

DUMMY_TEXT   = "Hello, world!"
DUMMY_PROMPT = "What is the capital of France?"


def _gen(model_id: str, **kw):
    """text-generation pipeline helper."""
    pipe = pipeline("text-generation", model=model_id, **kw)
    return pipe(DUMMY_PROMPT, max_new_tokens=32)


def _mask(model_id: str):
    """fill-mask pipeline helper."""
    pipe = pipeline("fill-mask", model=model_id)
    # Each tokeniser uses a different mask token
    mask = pipe.tokenizer.mask_token
    return pipe(f"The capital of France is {mask}.")


def _feat(model_id: str):
    """feature-extraction pipeline helper."""
    pipe = pipeline("feature-extraction", model=model_id)
    return pipe(DUMMY_TEXT)


def _seq2seq(model_id: str):
    """text2text-generation pipeline helper."""
    pipe = pipeline("text2text-generation", model=model_id)
    return pipe(DUMMY_PROMPT, max_new_tokens=32)


# ── 01-ai ──────────────────────────────────────────────────────────────────
# 01-ai/Yi-34B
_gen("01-ai/Yi-34B")

# 01-ai/Yi-34B-Chat
_gen("01-ai/Yi-34B-Chat")

# ── allenai ────────────────────────────────────────────────────────────────
# allenai/OLMo-7B
_gen("allenai/OLMo-7B")

# allenai/OLMo-7B-Instruct
_gen("allenai/OLMo-7B-Instruct")

# allenai/OLMo-7B-Twin-2T
_gen("allenai/OLMo-7B-Twin-2T")

# allenai/scibert_scivocab_cased
_mask("allenai/scibert_scivocab_cased")

# allenai/scibert_scivocab_uncased
_mask("allenai/scibert_scivocab_uncased")

# ── arcee-ai ───────────────────────────────────────────────────────────────
# arcee-ai/saul-zephyr-7b-slerp
_gen("arcee-ai/saul-zephyr-7b-slerp")

# ── bigcode ────────────────────────────────────────────────────────────────
# bigcode/santacoder
_gen("bigcode/santacoder")

# bigcode/starcoder
_gen("bigcode/starcoder")

# ── bigscience ─────────────────────────────────────────────────────────────
# bigscience/bloom-1b1
_gen("bigscience/bloom-1b1")

# bigscience/bloom-1b7
_gen("bigscience/bloom-1b7")

# bigscience/bloom-3b
_gen("bigscience/bloom-3b")

# bigscience/bloom-560m
_gen("bigscience/bloom-560m")

# bigscience/bloom-7b1
_gen("bigscience/bloom-7b1")

# bigscience/mt0-base
_seq2seq("bigscience/mt0-base")

# bigscience/mt0-small
_seq2seq("bigscience/mt0-small")

# ── cardiffnlp ─────────────────────────────────────────────────────────────
# cardiffnlp/twitter-roberta-base-sentiment-latest
pipe_sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
pipe_sentiment(DUMMY_TEXT)

# ── codellama ──────────────────────────────────────────────────────────────
# codellama/CodeLlama-13b-hf
_gen("codellama/CodeLlama-13b-hf")

# codellama/CodeLlama-13b-Instruct-hf
_gen("codellama/CodeLlama-13b-Instruct-hf")

# codellama/CodeLlama-34b-hf
_gen("codellama/CodeLlama-34b-hf")

# codellama/CodeLlama-34b-Instruct-hf
_gen("codellama/CodeLlama-34b-Instruct-hf")

# codellama/CodeLlama-70b-hf
_gen("codellama/CodeLlama-70b-hf")

# codellama/CodeLlama-70b-Instruct-hf
_gen("codellama/CodeLlama-70b-Instruct-hf")

# codellama/CodeLlama-7b-hf
_gen("codellama/CodeLlama-7b-hf")

# codellama/CodeLlama-7b-Instruct-hf
_gen("codellama/CodeLlama-7b-Instruct-hf")

# ── CohereForAI ────────────────────────────────────────────────────────────
# CohereForAI/c4ai-command-r7b-12-2024
_gen("CohereForAI/c4ai-command-r7b-12-2024")

# ── deepseek-ai ────────────────────────────────────────────────────────────
# deepseek-ai/deepseek-coder-33b-instruct
_gen("deepseek-ai/deepseek-coder-33b-instruct")

# deepseek-ai/deepseek-llm-67b-chat
_gen("deepseek-ai/deepseek-llm-67b-chat")

# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
_gen("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
_gen("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

# ── distilbert ─────────────────────────────────────────────────────────────
# distilbert/distilbert-base-uncased
_mask("distilbert/distilbert-base-uncased")

# distilbert/distilroberta-base
_mask("distilbert/distilroberta-base")

# ── facebook ───────────────────────────────────────────────────────────────
# facebook/bart-large
pipe_bart = pipeline("summarization", model="facebook/bart-large")
pipe_bart("Hugging Face is a technology company focused on NLP.", max_new_tokens=32)

# facebook/blenderbot-400M-distill
pipe_blender = pipeline("conversational", model="facebook/blenderbot-400M-distill")
pipe_blender(Conversation(DUMMY_PROMPT))

# ── FacebookAI ─────────────────────────────────────────────────────────────
# FacebookAI/roberta-base
_mask("FacebookAI/roberta-base")

# FacebookAI/roberta-large
_mask("FacebookAI/roberta-large")

# ── google-bert ────────────────────────────────────────────────────────────
# google-bert/bert-base-cased
_mask("google-bert/bert-base-cased")

# google-bert/bert-base-uncased
_mask("google-bert/bert-base-uncased")

# google-bert/bert-large-uncased
_mask("google-bert/bert-large-uncased")

# ── google-t5 ──────────────────────────────────────────────────────────────
# google-t5/t5-base
_seq2seq("google-t5/t5-base")

# google-t5/t5-large
_seq2seq("google-t5/t5-large")

# google-t5/t5-small
_seq2seq("google-t5/t5-small")

# ── google (text / seq2seq) ────────────────────────────────────────────────
# google/flan-t5-xxl
_seq2seq("google/flan-t5-xxl")

# google/gemma-2-9b
_gen("google/gemma-2-9b")

# google/gemma-2-9b-it
_gen("google/gemma-2-9b-it")

# google/gemma-2b-it
_gen("google/gemma-2b-it")

# google/gemma-3-12b-it
_gen("google/gemma-3-12b-it")

# google/gemma-3-27b-it
_gen("google/gemma-3-27b-it")

# google/gemma-7b-it
_gen("google/gemma-7b-it")

# google/pegasus-large
pipe_pegasus = pipeline("summarization", model="google/pegasus-large")
pipe_pegasus("Hugging Face is a technology company focused on NLP.", max_new_tokens=32)

# ── j-hartmann ─────────────────────────────────────────────────────────────
# j-hartmann/emotion-english-distilroberta-base
pipe_emotion = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
)
pipe_emotion(DUMMY_TEXT)

# ── lmsys ──────────────────────────────────────────────────────────────────
# lmsys/vicuna-13b-v1.5
_gen("lmsys/vicuna-13b-v1.5")

# lmsys/vicuna-7b-v1.5
_gen("lmsys/vicuna-7b-v1.5")

# ── meta-llama ─────────────────────────────────────────────────────────────
# meta-llama/Llama-2-13b-chat-hf
_gen("meta-llama/Llama-2-13b-chat-hf")

# meta-llama/Llama-2-70b-chat-hf
_gen("meta-llama/Llama-2-70b-chat-hf")

# meta-llama/Llama-2-7b-chat-hf
_gen("meta-llama/Llama-2-7b-chat-hf")

# meta-llama/Llama-3.1-405B-Instruct-FP8
_gen("meta-llama/Llama-3.1-405B-Instruct-FP8")

# meta-llama/Llama-3.1-70B
_gen("meta-llama/Llama-3.1-70B")

# meta-llama/Llama-3.1-70B-Instruct
_gen("meta-llama/Llama-3.1-70B-Instruct")

# meta-llama/Llama-3.2-1B
_gen("meta-llama/Llama-3.2-1B")

# meta-llama/Llama-3.2-1B-Instruct
_gen("meta-llama/Llama-3.2-1B-Instruct")

# meta-llama/Llama-3.2-3B
_gen("meta-llama/Llama-3.2-3B")

# meta-llama/Llama-3.2-3B-Instruct
_gen("meta-llama/Llama-3.2-3B-Instruct")

# meta-llama/Llama-3.3-70B-Instruct
_gen("meta-llama/Llama-3.3-70B-Instruct")

# meta-llama/Llama-4-Scout-17B-16E-Instruct
_gen("meta-llama/Llama-4-Scout-17B-16E-Instruct")

# meta-llama/Meta-Llama-3-70B
_gen("meta-llama/Meta-Llama-3-70B")

# meta-llama/Meta-Llama-3-70B-Instruct
_gen("meta-llama/Meta-Llama-3-70B-Instruct")

# meta-llama/Meta-Llama-3-8B
_gen("meta-llama/Meta-Llama-3-8B")

# meta-llama/Meta-Llama-3-8B-Instruct
_gen("meta-llama/Meta-Llama-3-8B-Instruct")

# meta-llama/Meta-Llama-3.1-8B
_gen("meta-llama/Meta-Llama-3.1-8B")

# meta-llama/Meta-Llama-3.1-8B-Instruct
_gen("meta-llama/Meta-Llama-3.1-8B-Instruct")

# ── microsoft (NLP) ────────────────────────────────────────────────────────
# microsoft/deberta-base
_mask("microsoft/deberta-base")

# microsoft/deberta-v3-base
_mask("microsoft/deberta-v3-base")

# microsoft/DialoGPT-medium
pipe_dialogpt = pipeline("text-generation", model="microsoft/DialoGPT-medium")
pipe_dialogpt(DUMMY_PROMPT, max_new_tokens=32)

# microsoft/Phi-3-medium-4k-instruct
_gen("microsoft/Phi-3-medium-4k-instruct")

# microsoft/Phi-3-mini-4k-instruct
_gen("microsoft/Phi-3-mini-4k-instruct")

# microsoft/phi-4
_gen("microsoft/phi-4")

# ── mistralai ──────────────────────────────────────────────────────────────
# mistralai/Codestral-22B-v0.1
_gen("mistralai/Codestral-22B-v0.1")

# mistralai/Ministral-8B-Instruct-2410
_gen("mistralai/Ministral-8B-Instruct-2410")

# mistralai/Mistral-7B-Instruct-v0.1
_gen("mistralai/Mistral-7B-Instruct-v0.1")

# mistralai/Mistral-7B-Instruct-v0.2
_gen("mistralai/Mistral-7B-Instruct-v0.2")

# mistralai/Mistral-7B-Instruct-v0.3
_gen("mistralai/Mistral-7B-Instruct-v0.3")

# mistralai/Mistral-7B-v0.3
_gen("mistralai/Mistral-7B-v0.3")

# mistralai/Mistral-Large-Instruct-2407
_gen("mistralai/Mistral-Large-Instruct-2407")

# mistralai/Mistral-Small-3.2-24B-Instruct-2506
_gen("mistralai/Mistral-Small-3.2-24B-Instruct-2506")

# mistralai/Mixtral-8x22B-Instruct-v0.1
_gen("mistralai/Mixtral-8x22B-Instruct-v0.1")

# mistralai/Mixtral-8x7B-Instruct-v0.1
_gen("mistralai/Mixtral-8x7B-Instruct-v0.1")

# mistralai/Mixtral-8x7B-v0.1
_gen("mistralai/Mixtral-8x7B-v0.1")

# ── NousResearch ───────────────────────────────────────────────────────────
# NousResearch/Nous-Capybara-7B-V1.9
_gen("NousResearch/Nous-Capybara-7B-V1.9")

# NousResearch/Nous-Hermes-llama-2-7b
_gen("NousResearch/Nous-Hermes-llama-2-7b")

# NousResearch/Nous-Hermes-Llama2-13b
_gen("NousResearch/Nous-Hermes-Llama2-13b")

# ── nreimers ───────────────────────────────────────────────────────────────
# nreimers/BERT-Tiny_L-2_H-128_A-2
_mask("nreimers/BERT-Tiny_L-2_H-128_A-2")

# nreimers/MiniLM-L6-H384-uncased
_feat("nreimers/MiniLM-L6-H384-uncased")

# ── NumbersStation ─────────────────────────────────────────────────────────
# NumbersStation/nsql-llama-2-7B
_gen("NumbersStation/nsql-llama-2-7B")

# ── nvidia ─────────────────────────────────────────────────────────────────
# nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
_gen("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5")

# ── Open-Orca ──────────────────────────────────────────────────────────────
# Open-Orca/Mistral-7B-OpenOrca
_gen("Open-Orca/Mistral-7B-OpenOrca")

# ── openai (HF namespace, text generation) ─────────────────────────────────
# openai/gpt-oss-120b
_gen("openai/gpt-oss-120b")

# openai/gpt-oss-20b
_gen("openai/gpt-oss-20b")

# ── openchat ───────────────────────────────────────────────────────────────
# openchat/openchat-3.5-1210
_gen("openchat/openchat-3.5-1210")

# ── prajjwal1 ──────────────────────────────────────────────────────────────
# prajjwal1/bert-tiny
_mask("prajjwal1/bert-tiny")

# ── Qwen ───────────────────────────────────────────────────────────────────
# Qwen/Qwen1.5-0.5B-Chat
_gen("Qwen/Qwen1.5-0.5B-Chat")

# Qwen/Qwen1.5-1.8B-Chat
_gen("Qwen/Qwen1.5-1.8B-Chat")

# Qwen/Qwen1.5-110B-Chat
_gen("Qwen/Qwen1.5-110B-Chat")

# Qwen/Qwen1.5-14B-Chat
_gen("Qwen/Qwen1.5-14B-Chat")

# Qwen/Qwen1.5-32B-Chat
_gen("Qwen/Qwen1.5-32B-Chat")

# Qwen/Qwen1.5-4B-Chat
_gen("Qwen/Qwen1.5-4B-Chat")

# Qwen/Qwen1.5-72B-Chat
_gen("Qwen/Qwen1.5-72B-Chat")

# Qwen/Qwen1.5-7B-Chat
_gen("Qwen/Qwen1.5-7B-Chat")

# Qwen/Qwen2-72B-Instruct
_gen("Qwen/Qwen2-72B-Instruct")

# Qwen/Qwen2-7B-Instruct
_gen("Qwen/Qwen2-7B-Instruct")

# Qwen/Qwen2.5-32B
_gen("Qwen/Qwen2.5-32B")

# Qwen/Qwen3-4B
_gen("Qwen/Qwen3-4B")

# Qwen/Qwen3-Next-80B-A3B-Instruct
_gen("Qwen/Qwen3-Next-80B-A3B-Instruct")

# Qwen/QwQ-32B
_gen("Qwen/QwQ-32B")

# ── stabilityai ────────────────────────────────────────────────────────────
# stabilityai/stablelm-tuned-alpha-3b
_gen("stabilityai/stablelm-tuned-alpha-3b")

# ── tiiuae ─────────────────────────────────────────────────────────────────
# tiiuae/falcon-7b-instruct
_gen("tiiuae/falcon-7b-instruct")

# ── TinyLlama ──────────────────────────────────────────────────────────────
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
_gen("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ── togethercomputer ───────────────────────────────────────────────────────
# togethercomputer/RedPajama-INCITE-7B-Chat
_gen("togethercomputer/RedPajama-INCITE-7B-Chat")

# togethercomputer/RedPajama-INCITE-Chat-3B-v1
_gen("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
