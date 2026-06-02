from transformers import AutoModelForCausalLM, AutoTokenizer
import nemo.collections.asr as nemo_asr

# Load the GPT-OSS-120B model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-120b")

# Load the NeMo ASR model for speech-to-text transcription
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
