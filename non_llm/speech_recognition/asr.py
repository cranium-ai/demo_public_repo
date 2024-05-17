import torch
import librosa
import argparse
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the model and processor from Hugging Face
model_name = "openai/whisper-large-v2"  # Replace with the actual model name if needed
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Function to load and preprocess the audio file
def load_audio(file_path):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=16000)
    return audio, sample_rate

# Function to perform speech recognition
def transcribe_audio(audio, sample_rate):
    # Process the audio file
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

    # Generate transcription
    with torch.no_grad():
        logits = model.generate(inputs.input_features)

    # Decode the logits
    transcription = processor.batch_decode(logits, skip_special_tokens=True)[0]

    return transcription

# Main function
def main(audio_file_path):
    # Load and preprocess the audio file
    audio, sample_rate = load_audio(audio_file_path)

    # Perform speech recognition
    transcription = transcribe_audio(audio, sample_rate)

    # Print the transcription
    print("Transcription:", transcription)

# Replace 'path_to_your_audio_file.wav' with the path to your audio file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe an audio file')
    parser.add_argument('audio_file_path', help='Path to the audio file to be transcribed')
    args = parser.parse_args()

    main(args.audio_file_path)
