# =============================================================================
# 9.  HUGGING FACE  (all model-ids that contain '/')  – alphabetical
# =============================================================================

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Helper: generic text-generation pipeline
def _hf_text_gen(model_id: str, **kwargs):
    pipe = pipeline("text-generation", model=model_id, **kwargs)
    return pipe(DUMMY_PROMPT, max_new_tokens=32)

# Helper: generic fill-mask / feature-extraction pipeline
def _hf_pipe(task: str, model_id: str, **kwargs):
    pipe = pipeline(task, model=model_id, **kwargs)
    return pipe(DUMMY_TEXT)

# ── 01-ai ──────────────────────────────────────────────────────────────────
# 01-ai/Yi-34B
_hf_text_gen("01-ai/Yi-34B")

# 01-ai/Yi-34B-Chat
_hf_text_gen("01-ai/Yi-34B-Chat")

# ── allenai ────────────────────────────────────────────────────────────────
# allenai/OLMo-7B
_hf_text_gen("allenai/OLMo-7B")

# allenai/OLMo-7B-Instruct
_hf_text_gen("allenai/OLMo-7B-Instruct")

# allenai/OLMo-7B-Twin-2T
_hf_text_gen("allenai/OLMo-7B-Twin-2T")

# allenai/scibert_scivocab_cased
_hf_pipe("fill-mask", "allenai/scibert_scivocab_cased")

# allenai/scibert_scivocab_uncased
_hf_pipe("fill-mask", "allenai/scibert_scivocab_uncased")

# ── arcee-ai ───────────────────────────────────────────────────────────────
# arcee-ai/saul-zephyr-7b-slerp
_hf_text_gen("arcee-ai/saul-zephyr-7b-slerp")

# ── bigcode ────────────────────────────────────────────────────────────────
# bigcode/santacoder
_hf_text_gen("bigcode/santacoder")

# bigcode/starcoder
_hf_text_gen("bigcode/starcoder")

# ── bigscience ─────────────────────────────────────────────────────────────
# bigscience/bloom-1b1
_hf_text_gen("bigscience/bloom-1b1")

# bigscience/bloom-1b7
_hf_text_gen("bigscience/bloom-1b7")

# bigscience/bloom-3b
_hf_text_gen("bigscience/bloom-3b")

# bigscience/bloom-560m
_hf_text_gen("bigscience/bloom-560m")

# bigscience/bloom-7b1
_hf_text_gen("bigscience/bloom-7b1")

# bigscience/mt0-base
pipe_mt0_base = pipeline("text2text-generation", model="bigscience/mt0-base")
pipe_mt0_base(DUMMY_PROMPT, max_new_tokens=32)

# bigscience/mt0-small
pipe_mt0_small = pipeline("text2text-generation", model="bigscience/mt0-small")
pipe_mt0_small(DUMMY_PROMPT, max_new_tokens=32)

# ── cardiffnlp ─────────────────────────────────────────────────────────────
# cardiffnlp/twitter-roberta-base-sentiment-latest
pipe_sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
pipe_sentiment(DUMMY_TEXT)

# ── codellama ──────────────────────────────────────────────────────────────
# codellama/CodeLlama-13b-hf
_hf_text_gen("codellama/CodeLlama-13b-hf")

# codellama/CodeLlama-13b-Instruct-hf
_hf_text_gen("codellama/CodeLlama-13b-Instruct-hf")

# codellama/CodeLlama-34b-hf
_hf_text_gen("codellama/CodeLlama-34b-hf")

# codellama/CodeLlama-34b-Instruct-hf
_hf_text_gen("codellama/CodeLlama-34b-Instruct-hf")

# codellama/CodeLlama-70b-hf
_hf_text_gen("codellama/CodeLlama-70b-hf")

# codellama/CodeLlama-70b-Instruct-hf
_hf_text_gen("codellama/CodeLlama-70b-Instruct-hf")

# codellama/CodeLlama-7b-hf
_hf_text_gen("codellama/CodeLlama-7b-hf")

# codellama/CodeLlama-7b-Instruct-hf
_hf_text_gen("codellama/CodeLlama-7b-Instruct-hf")

# ── CohereForAI ────────────────────────────────────────────────────────────
# CohereForAI/c4ai-command-r7b-12-2024
_hf_text_gen("CohereForAI/c4ai-command-r7b-12-2024")

# ── deepseek-ai ────────────────────────────────────────────────────────────
# deepseek-ai/deepseek-coder-33b-instruct
_hf_text_gen("deepseek-ai/deepseek-coder-33b-instruct")

# deepseek-ai/deepseek-llm-67b-chat
_hf_text_gen("deepseek-ai/deepseek-llm-67b-chat")

# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
_hf_text_gen("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
_hf_text_gen("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

# ── distilbert ─────────────────────────────────────────────────────────────
# distilbert/distilbert-base-uncased
_hf_pipe("fill-mask", "distilbert/distilbert-base-uncased")

# distilbert/distilroberta-base
_hf_pipe("fill-mask", "distilbert/distilroberta-base")

# ── facebook ───────────────────────────────────────────────────────────────
# facebook/bart-large
pipe_bart = pipeline("summarization", model="facebook/bart-large")
pipe_bart("Hugging Face is a technology company.", max_new_tokens=32)

# facebook/blenderbot-400M-distill
pipe_blender = pipeline("conversational", model="facebook/blenderbot-400M-distill")
from transformers import Conversation
pipe_blender(Conversation(DUMMY_PROMPT))

# facebook/regnet-x-016
pipe_regnet_016 = pipeline("image-classification", model="facebook/regnet-x-016")
pipe_regnet_016(DUMMY_IMAGE_PATH)

# facebook/regnet-x-032
pipe_regnet_032 = pipeline("image-classification", model="facebook/regnet-x-032")
pipe_regnet_032(DUMMY_IMAGE_PATH)

# facebook/regnet-x-040
pipe_regnet_040 = pipeline("image-classification", model="facebook/regnet-x-040")
pipe_regnet_040(DUMMY_IMAGE_PATH)

# facebook/regnet-x-064
pipe_regnet_064 = pipeline("image-classification", model="facebook/regnet-x-064")
pipe_regnet_064(DUMMY_IMAGE_PATH)

# facebook/regnet-x-080
pipe_regnet_080 = pipeline("image-classification", model="facebook/regnet-x-080")
pipe_regnet_080(DUMMY_IMAGE_PATH)

# facebook/regnet-x-120
pipe_regnet_120 = pipeline("image-classification", model="facebook/regnet-x-120")
pipe_regnet_120(DUMMY_IMAGE_PATH)

# ── FacebookAI ─────────────────────────────────────────────────────────────
# FacebookAI/roberta-base
_hf_pipe("fill-mask", "FacebookAI/roberta-base")

# FacebookAI/roberta-large
_hf_pipe("fill-mask", "FacebookAI/roberta-large")

# ── google ─────────────────────────────────────────────────────────────────
# google-bert/bert-base-cased
_hf_pipe("fill-mask", "google-bert/bert-base-cased")

# google-bert/bert-base-uncased
_hf_pipe("fill-mask", "google-bert/bert-base-uncased")

# google-bert/bert-large-uncased
_hf_pipe("fill-mask", "google-bert/bert-large-uncased")

# google-t5/t5-base
pipe_t5_base = pipeline("text2text-generation", model="google-t5/t5-base")
pipe_t5_base(DUMMY_PROMPT, max_new_tokens=32)

# google-t5/t5-large
pipe_t5_large = pipeline("text2text-generation", model="google-t5/t5-large")
pipe_t5_large(DUMMY_PROMPT, max_new_tokens=32)

# google-t5/t5-small
pipe_t5_small = pipeline("text2text-generation", model="google-t5/t5-small")
pipe_t5_small(DUMMY_PROMPT, max_new_tokens=32)

# google/efficientnet-b0
pipe_effnet_b0 = pipeline("image-classification", model="google/efficientnet-b0")
pipe_effnet_b0(DUMMY_IMAGE_PATH)

# google/efficientnet-b7
pipe_effnet_b7 = pipeline("image-classification", model="google/efficientnet-b7")
pipe_effnet_b7(DUMMY_IMAGE_PATH)

# google/flan-t5-xxl
pipe_flan = pipeline("text2text-generation", model="google/flan-t5-xxl")
pipe_flan(DUMMY_PROMPT, max_new_tokens=32)

# google/gemma-2-9b
_hf_text_gen("google/gemma-2-9b")

# google/gemma-2-9b-it
_hf_text_gen("google/gemma-2-9b-it")

# google/gemma-2b-it
_hf_text_gen("google/gemma-2b-it")

# google/gemma-3-12b-it
_hf_text_gen("google/gemma-3-12b-it")

# google/gemma-3-27b-it
_hf_text_gen("google/gemma-3-27b-it")

# google/gemma-7b-it
_hf_text_gen("google/gemma-7b-it")

# google/pegasus-large
pipe_pegasus = pipeline("summarization", model="google/pegasus-large")
pipe_pegasus("Hugging Face is a technology company.", max_new_tokens=32)

# ── HuggingFaceM4 ──────────────────────────────────────────────────────────
# HuggingFaceM4/idefics-80b  (multimodal; image+text)
from transformers import IdeficsForVisionText2Text, AutoProcessor
processor_idefics = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-80b")
model_idefics = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-80b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs_idefics = processor_idefics(
    text=[DUMMY_PROMPT], return_tensors="pt"
)
model_idefics.generate(**inputs_idefics, max_new_tokens=32)

# HuggingFaceM4/idefics-9b
processor_idefics_9b = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")
model_idefics_9b = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-9b",
    torch_dtype=torch.bfloat16,
)
inputs_idefics_9b = processor_idefics_9b(
    text=[DUMMY_PROMPT], return_tensors="pt"
)
model_idefics_9b.generate(**inputs_idefics_9b, max_new_tokens=32)

# HuggingFaceM4/idefics-9b-instruct
processor_idefics_9b_inst = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics-9b-instruct"
)
model_idefics_9b_inst = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-9b-instruct",
    torch_dtype=torch.bfloat16,
)
inputs_idefics_9b_inst = processor_idefics_9b_inst(
    text=[DUMMY_PROMPT], return_tensors="pt"
)
model_idefics_9b_inst.generate(**inputs_idefics_9b_inst, max_new_tokens=32)

# ── j-hartmann ─────────────────────────────────────────────────────────────
# j-hartmann/emotion-english-distilroberta-base
pipe_emotion = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
)
pipe_emotion(DUMMY_TEXT)

# ── llava-hf ───────────────────────────────────────────────────────────────
# llava-hf/llava-1.5-13b-hf
from transformers import LlavaForConditionalGeneration, AutoProcessor as LlavaProcessor
processor_llava = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
model_llava = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16
)
inputs_llava = processor_llava(
    text=DUMMY_PROMPT, return_tensors="pt"
)
model_llava.generate(**inputs_llava, max_new_tokens=32)

# llava/hf_llava-v1.6-mistral-7b-hf
processor_llava16 = LlavaProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
model_llava16 = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16
)
inputs_llava16 = processor_llava16(
    text=DUMMY_PROMPT, return_tensors="pt"
)
model_llava16.generate(**inputs_llava16, max_new_tokens=32)

# ── lmsys ──────────────────────────────────────────────────────────────────
# lmsys/vicuna-13b-v1.5
_hf_text_gen("lmsys/vicuna-13b-v1.5")

# lmsys/vicuna-7b-v1.5
_hf_text_gen("lmsys/vicuna-7b-v1.5")

# ── meta-llama ─────────────────────────────────────────────────────────────
# meta-llama/Llama-2-13b-chat-hf
_hf_text_gen("meta-llama/Llama-2-13b-chat-hf")

# meta-llama/Llama-2-70b-chat-hf
_hf_text_gen("meta-llama/Llama-2-70b-chat-hf")

# meta-llama/Llama-2-7b-chat-hf
_hf_text_gen("meta-llama/Llama-2-7b-chat-hf")

# meta-llama/Llama-3.1-405B-Instruct-FP8
_hf_text_gen("meta-llama/Llama-3.1-405B-Instruct-FP8")

# meta-llama/Llama-3.1-70B
_hf_text_gen("meta-llama/Llama-3.1-70B")

# meta-llama/Llama-3.1-70B-Instruct
_hf_text_gen("meta-llama/Llama-3.1-70B-Instruct")

# meta-llama/Llama-3.2-1B
_hf_text_gen("meta-llama/Llama-3.2-1B")

# meta-llama/Llama-3.2-1B-Instruct
_hf_text_gen("meta-llama/Llama-3.2-1B-Instruct")

# meta-llama/Llama-3.2-3B
_hf_text_gen("meta-llama/Llama-3.2-3B")

# meta-llama/Llama-3.2-3B-Instruct
_hf_text_gen("meta-llama/Llama-3.2-3B-Instruct")

# meta-llama/Llama-3.3-70B-Instruct
_hf_text_gen("meta-llama/Llama-3.3-70B-Instruct")

# meta-llama/Llama-4-Scout-17B-16E-Instruct
_hf_text_gen("meta-llama/Llama-4-Scout-17B-16E-Instruct")

# meta-llama/Meta-Llama-3-70B
_hf_text_gen("meta-llama/Meta-Llama-3-70B")

# meta-llama/Meta-Llama-3-70B-Instruct
_hf_text_gen("meta-llama/Meta-Llama-3-70B-Instruct")

# meta-llama/Meta-Llama-3-8B
_hf_text_gen("meta-llama/Meta-Llama-3-8B")

# meta-llama/Meta-Llama-3-8B-Instruct
_hf_text_gen("meta-llama/Meta-Llama-3-8B-Instruct")

# meta-llama/Meta-Llama-3.1-8B
_hf_text_gen("meta-llama/Meta-Llama-3.1-8B")

# meta-llama/Meta-Llama-3.1-8B-Instruct
_hf_text_gen("meta-llama/Meta-Llama-3.1-8B-Instruct")

# ── microsoft ──────────────────────────────────────────────────────────────
# microsoft/deberta-base
_hf_pipe("fill-mask", "microsoft/deberta-base")

# microsoft/deberta-v3-base
_hf_pipe("fill-mask", "microsoft/deberta-v3-base")

# microsoft/DialoGPT-medium
pipe_dialogpt = pipeline("text-generation", model="microsoft/DialoGPT-medium")
pipe_dialogpt(DUMMY_PROMPT, max_new_tokens=32)

# microsoft/focalnet-tiny-lrf
pipe_focalnet = pipeline("image-classification", model="microsoft/focalnet-tiny-lrf")
pipe_focalnet(DUMMY_IMAGE_PATH)

# microsoft/Phi-3-medium-4k-instruct
_hf_text_gen("microsoft/Phi-3-medium-4k-instruct")

# microsoft/Phi-3-mini-4k-instruct
_hf_text_gen("microsoft/Phi-3-mini-4k-instruct")

# microsoft/phi-4
_hf_text_gen("microsoft/phi-4")

# microsoft/resnet-101
pipe_ms_rn101 = pipeline("image-classification", model="microsoft/resnet-101")
pipe_ms_rn101(DUMMY_IMAGE_PATH)

# microsoft/resnet-152
pipe_ms_rn152 = pipeline("image-classification", model="microsoft/resnet-152")
pipe_ms_rn152(DUMMY_IMAGE_PATH)

# microsoft/resnet-18
pipe_ms_rn18 = pipeline("image-classification", model="microsoft/resnet-18")
pipe_ms_rn18(DUMMY_IMAGE_PATH)

# microsoft/resnet-50
pipe_ms_rn50 = pipeline("image-classification", model="microsoft/resnet-50")
pipe_ms_rn50(DUMMY_IMAGE_PATH)

# ── mistralai ──────────────────────────────────────────────────────────────
# mistralai/Codestral-22B-v0.1
_hf_text_gen("mistralai/Codestral-22B-v0.1")

# mistralai/Ministral-8B-Instruct-2410
_hf_text_gen("mistralai/Ministral-8B-Instruct-2410")

# mistralai/Mistral-7B-Instruct-v0.1
_hf_text_gen("mistralai/Mistral-7B-Instruct-v0.1")

# mistralai/Mistral-7B-Instruct-v0.2
_hf_text_gen("mistralai/Mistral-7B-Instruct-v0.2")

# mistralai/Mistral-7B-Instruct-v0.3
_hf_text_gen("mistralai/Mistral-7B-Instruct-v0.3")

# mistralai/Mistral-7B-v0.3
_hf_text_gen("mistralai/Mistral-7B-v0.3")

# mistralai/Mistral-Large-Instruct-2407
_hf_text_gen("mistralai/Mistral-Large-Instruct-2407")

# mistralai/Mistral-Small-3.2-24B-Instruct-2506
_hf_text_gen("mistralai/Mistral-Small-3.2-24B-Instruct-2506")

# mistralai/Mixtral-8x22B-Instruct-v0.1
_hf_text_gen("mistralai/Mixtral-8x22B-Instruct-v0.1")

# mistralai/Mixtral-8x7B-Instruct-v0.1
_hf_text_gen("mistralai/Mixtral-8x7B-Instruct-v0.1")

# mistralai/Mixtral-8x7B-v0.1
_hf_text_gen("mistralai/Mixtral-8x7B-v0.1")

# ── NousResearch ───────────────────────────────────────────────────────────
# NousResearch/Nous-Capybara-7B-V1.9
_hf_text_gen("NousResearch/Nous-Capybara-7B-V1.9")

# NousResearch/Nous-Hermes-llama-2-7b
_hf_text_gen("NousResearch/Nous-Hermes-llama-2-7b")

# NousResearch/Nous-Hermes-Llama2-13b
_hf_text_gen("NousResearch/Nous-Hermes-Llama2-13b")

# ── nreimers ───────────────────────────────────────────────────────────────
# nreimers/BERT-Tiny_L-2_H-128_A-2
_hf_pipe("fill-mask", "nreimers/BERT-Tiny_L-2_H-128_A-2")

# nreimers/MiniLM-L6-H384-uncased
_hf_pipe("feature-extraction", "nreimers/MiniLM-L6-H384-uncased")

# ── NumbersStation ─────────────────────────────────────────────────────────
# NumbersStation/nsql-llama-2-7B
_hf_text_gen("NumbersStation/nsql-llama-2-7B")

# ── nvidia ─────────────────────────────────────────────────────────────────
# nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
_hf_text_gen("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5")

# nvidia/parakeet-tdt-0.6b-v3  (ASR model)
pipe_parakeet = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")
# pipe_parakeet("audio.wav")  # provide a real audio file at runtime

# ── Open-Orca ──────────────────────────────────────────────────────────────
# Open-Orca/Mistral-7B-OpenOrca
_hf_text_gen("Open-Orca/Mistral-7B-OpenOrca")

# ── openai (HF namespace) ──────────────────────────────────────────────────
# openai/gpt-oss-120b
_hf_text_gen("openai/gpt-oss-120b")

# openai/gpt-oss-20b
_hf_text_gen("openai/gpt-oss-20b")

# openai/whisper-base  (ASR)
pipe_whisper_base = pipeline("automatic-speech-recognition", model="openai/whisper-base")
# pipe_whisper_base("audio.wav")

# openai/whisper-large-v3-turbo
pipe_whisper_large = pipeline(
    "automatic-speech-recognition", model="openai/whisper-large-v3-turbo"
)
# pipe_whisper_large("audio.wav")

# ── openchat ───────────────────────────────────────────────────────────────
# openchat/openchat-3.5-1210
_hf_text_gen("openchat/openchat-3.5-1210")

# ── prajjwal1 ──────────────────────────────────────────────────────────────
# prajjwal1/bert-tiny
_hf_pipe("fill-mask", "prajjwal1/bert-tiny")

# ── Qwen ───────────────────────────────────────────────────────────────────
# Qwen/Qwen1.5-0.5B-Chat
_hf_text_gen("Qwen/Qwen1.5-0.5B-Chat")

# Qwen/Qwen1.5-1.8B-Chat
_hf_text_gen("Qwen/Qwen1.5-1.8B-Chat")

# Qwen/Qwen1.5-110B-Chat
_hf_text_gen("Qwen/Qwen1.5-110B-Chat")

# Qwen/Qwen1.5-14B-Chat
_hf_text_gen("Qwen/Qwen1.5-14B-Chat")

# Qwen/Qwen1.5-32B-Chat
_hf_text_gen("Qwen/Qwen1.5-32B-Chat")

# Qwen/Qwen1.5-4B-Chat
_hf_text_gen("Qwen/Qwen1.5-4B-Chat")

# Qwen/Qwen1.5-72B-Chat
_hf_text_gen("Qwen/Qwen1.5-72B-Chat")

# Qwen/Qwen1.5-7B-Chat
_hf_text_gen("Qwen/Qwen1.5-7B-Chat")

# Qwen/Qwen2-72B-Instruct
_hf_text_gen("Qwen/Qwen2-72B-Instruct")

# Qwen/Qwen2-7B-Instruct
_hf_text_gen("Qwen/Qwen2-7B-Instruct")

# Qwen/Qwen2.5-32B
_hf_text_gen("Qwen/Qwen2.5-32B")

# Qwen/Qwen3-4B
_hf_text_gen("Qwen/Qwen3-4B")

# Qwen/Qwen3-Next-80B-A3B-Instruct
_hf_text_gen("Qwen/Qwen3-Next-80B-A3B-Instruct")

# Qwen/QwQ-32B
_hf_text_gen("Qwen/QwQ-32B")

# ── stabilityai ────────────────────────────────────────────────────────────
# stabilityai/stablelm-tuned-alpha-3b
_hf_text_gen("stabilityai/stablelm-tuned-alpha-3b")

# ── tiiuae ─────────────────────────────────────────────────────────────────
# tiiuae/falcon-7b-instruct
_hf_text_gen("tiiuae/falcon-7b-instruct")

# ── timm (namespace/repo models) ───────────────────────────────────────────
# timm/eva_giant_patch14_224.clip_ft_in1k
model_eva = timm.create_model(
    "timm/eva_giant_patch14_224.clip_ft_in1k", pretrained=False
)
model_eva.eval()
model_eva(_dummy_tensor)

# timm/inception_v4.tf_in1k
model_iv4 = timm.create_model("timm/inception_v4.tf_in1k", pretrained=False)
model_iv4.eval()
model_iv4(torch.randn(1, 3, 299, 299))

# timm/levit_128s.fb_dist_in1k
model_levit = timm.create_model("timm/levit_128s.fb_dist_in1k", pretrained=False)
model_levit.eval()
model_levit(_dummy_tensor)

# timm/tf_efficientnet_lite0.in1k
model_effnetlite = timm.create_model(
    "timm/tf_efficientnet_lite0.in1k", pretrained=False
)
model_effnetlite.eval()
model_effnetlite(_dummy_tensor)

# ── TinyLlama ──────────────────────────────────────────────────────────────
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
_hf_text_gen("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ── togethercomputer ───────────────────────────────────────────────────────
# togethercomputer/RedPajama-INCITE-7B-Chat
_hf_text_gen("togethercomputer/RedPajama-INCITE-7B-Chat")

# togethercomputer/RedPajama-INCITE-Chat-3B-v1
_hf_text_gen("togethercomputer/RedPajama-INCITE-Chat-3B-v1")

# ── unsloth ────────────────────────────────────────────────────────────────
# unsloth/DeepSeek-R1-GGUF  – use llama-cpp-python for GGUF models
from llama_cpp import Llama
llm_gguf = Llama.from_pretrained(
    repo_id="unsloth/DeepSeek-R1-GGUF",
    filename="*Q4_K_M.gguf",    # pick a quantisation file
)
llm_gguf(DUMMY_PROMPT, max_tokens=32)