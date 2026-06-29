"""
10_hf_multimodal_misc.py
========================
Hugging Face multimodal models, timm models referenced by HF repo-id,
and the GGUF model (llama-cpp-python) — alphabetical.

Install:
    pip install transformers torch accelerate timm llama-cpp-python
"""

import torch
import timm
from transformers import (
    AutoProcessor,
    IdeficsForVisionText2Text,
    LlavaForConditionalGeneration,
)

DUMMY_PROMPT = "What is the capital of France?"
_t224 = torch.randn(1, 3, 224, 224)
_t299 = torch.randn(1, 3, 299, 299)

# ── HuggingFaceM4 — IDEFICS multimodal ───────────────────────────────────
# HuggingFaceM4/idefics-80b
processor_80b = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-80b")
model_80b = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-80b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs_80b = processor_80b(text=[DUMMY_PROMPT], return_tensors="pt")
model_80b.generate(**inputs_80b, max_new_tokens=32)

# HuggingFaceM4/idefics-9b
processor_9b = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")
model_9b = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-9b",
    torch_dtype=torch.bfloat16,
)
inputs_9b = processor_9b(text=[DUMMY_PROMPT], return_tensors="pt")
model_9b.generate(**inputs_9b, max_new_tokens=32)

# HuggingFaceM4/idefics-9b-instruct
processor_9b_inst = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics-9b-instruct"
)
model_9b_inst = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-9b-instruct",
    torch_dtype=torch.bfloat16,
)
inputs_9b_inst = processor_9b_inst(text=[DUMMY_PROMPT], return_tensors="pt")
model_9b_inst.generate(**inputs_9b_inst, max_new_tokens=32)

# ── llava-hf — LLaVA multimodal ───────────────────────────────────────────
# llava-hf/llava-1.5-13b-hf
processor_llava = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
model_llava = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16
)
inputs_llava = processor_llava(text=DUMMY_PROMPT, return_tensors="pt")
model_llava.generate(**inputs_llava, max_new_tokens=32)

# llava/hf_llava-v1.6-mistral-7b-hf
# (canonical HF repo: llava-hf/llava-v1.6-mistral-7b-hf)
processor_llava16 = AutoProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
model_llava16 = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16
)
inputs_llava16 = processor_llava16(text=DUMMY_PROMPT, return_tensors="pt")
model_llava16.generate(**inputs_llava16, max_new_tokens=32)

# ── timm HF-hosted models (contain '/') ──────────────────────────────────
# timm/eva_giant_patch14_224.clip_ft_in1k
model_eva = timm.create_model(
    "timm/eva_giant_patch14_224.clip_ft_in1k", pretrained=False
)
model_eva.eval()
model_eva(_t224)

# timm/inception_v4.tf_in1k
model_iv4 = timm.create_model("timm/inception_v4.tf_in1k", pretrained=False)
model_iv4.eval()
model_iv4(_t299)

# timm/levit_128s.fb_dist_in1k
model_levit = timm.create_model("timm/levit_128s.fb_dist_in1k", pretrained=False)
model_levit.eval()
model_levit(_t224)

# timm/tf_efficientnet_lite0.in1k
model_effnetlite = timm.create_model(
    "timm/tf_efficientnet_lite0.in1k", pretrained=False
)
model_effnetlite.eval()
model_effnetlite(_t224)

# ── unsloth — GGUF quantised model via llama-cpp-python ──────────────────
# unsloth/DeepSeek-R1-GGUF
from llama_cpp import Llama

llm_gguf = Llama.from_pretrained(
    repo_id="unsloth/DeepSeek-R1-GGUF",
    filename="*Q4_K_M.gguf",   # selects a Q4_K_M quantisation file
)
llm_gguf(DUMMY_PROMPT, max_tokens=32)
