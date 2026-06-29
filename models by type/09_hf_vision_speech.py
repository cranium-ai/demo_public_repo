"""
09_hf_vision_speech.py
======================
Hugging Face vision and automatic-speech-recognition models (contain '/')
— alphabetical.

Install:
    pip install transformers torch torchvision Pillow
"""

from transformers import pipeline

DUMMY_IMAGE_PATH = "sample.jpg"   # provide a real image at runtime

# ── facebook (vision) ──────────────────────────────────────────────────────
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

# ── google (vision) ────────────────────────────────────────────────────────
# google/efficientnet-b0
pipe_effnet_b0 = pipeline("image-classification", model="google/efficientnet-b0")
pipe_effnet_b0(DUMMY_IMAGE_PATH)

# google/efficientnet-b7
pipe_effnet_b7 = pipeline("image-classification", model="google/efficientnet-b7")
pipe_effnet_b7(DUMMY_IMAGE_PATH)

# ── microsoft (vision) ─────────────────────────────────────────────────────
# microsoft/focalnet-tiny-lrf
pipe_focalnet = pipeline("image-classification", model="microsoft/focalnet-tiny-lrf")
pipe_focalnet(DUMMY_IMAGE_PATH)

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

# ── nvidia (speech) ────────────────────────────────────────────────────────
# nvidia/parakeet-tdt-0.6b-v3
pipe_parakeet = pipeline(
    "automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3"
)
# pipe_parakeet("audio.wav")   # provide a real audio file at runtime

# ── openai (speech — HF namespace) ────────────────────────────────────────
# openai/whisper-base
pipe_whisper_base = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base"
)
# pipe_whisper_base("audio.wav")

# openai/whisper-large-v3-turbo
pipe_whisper_large = pipeline(
    "automatic-speech-recognition", model="openai/whisper-large-v3-turbo"
)
# pipe_whisper_large("audio.wav")
