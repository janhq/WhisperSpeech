import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import gradio as gr
import torch
import torchaudio
import whisper
from transformers import pipeline

from ichigo_whisper.demo.utils import load_model

# ! Ichigo Whisper
ichigo_name = "homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth"
model_size = "merge-medium-vi-2d-2560c-dim64"
ichigo_prompt = """
You are a professional transcriptionist fluent in both Vietnamese and English. 
You are tasked with transcribing an audio recording in which the speaker may switch between Vietnamese and English or speak entirely in one of these two languages. 
No other languages are present in the recording. The speaker may have a regional accent in either language. Your goal is to provide an accurate transcription of all spoken content in the appropriate language. 
If a translated section appears nonsensical or inaccurate, feel free to revise it to ensure clarity and correctness.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

ichigo_model = load_model(ref=ichigo_name, size=model_size)
ichigo_model.ensure_whisper(device, prompt=ichigo_prompt)
ichigo_model.to(device)

# ! Whisper Medium
whisper_model_name = "medium"
whisper_model = whisper.load_model(whisper_model_name)
whisper_model.to(device)

# ! PhoWhisper Medium
phowhisper = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-medium",
    device=device,
)


def transcribe_ichigo(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )
    wav, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    transcribe = ichigo_model.inference(wav.to(device))
    return transcribe[0].text


def transcribe_whisper(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )
    wav1, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav1 = torchaudio.functional.resample(wav1, sr, 16000)
    audio_sample = wav1.squeeze().float().numpy()
    transcribe = whisper_model.transcribe(
        audio_sample,
        task="transcribe",
        fp16=False,
    )
    return transcribe["text"]


def transcribe_phowhisper(inputs):
    wav2, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav2 = torchaudio.functional.resample(wav2, sr, 16000)
    audio_sample = wav2.squeeze().float().numpy()
    return phowhisper(audio_sample)["text"]


with gr.Blocks(title="Ichigo Whisper", theme="allenai/gradio-theme") as interface:
    gr.Markdown("# 🍰 Ichigo Whisper")
    gr.Markdown(
        "Ichigo Whisper is a compact (22M parameters), open-source quantizer for the Whisper-medium model, designed to enhance performance on low-resource languages with minimal impact on its original English capabilities. Unlike models that output continuous embeddings, Ichigo Whisper compresses speech into discrete tokens, making it more compatible with large language models (LLMs) for immediate speech understanding. This quantized version of Whisper-medium has been trained on over ~400 hours of English data and ~1000 hours of Vietnamese data."
    )
    gr.Markdown("Ichigo Whisper is a key component of the Ichigo v0.5 family.")
    gr.Markdown(
        "For more details, please refer to our official [blog post.](https://huggingface.co/homebrewltd/Ichigo-whisper-v0.1)"
    )
    with gr.Row():
        # Audio input section
        audio_input = gr.Audio(
            sources=["microphone", "upload"], type="filepath", label="Audio Input"
        )

    with gr.Row():
        # Ichigo Model Column
        with gr.Column():
            gr.Markdown("### Ichigo Whisper")
            ichigo_output = gr.TextArea(
                label="Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            ichigo_btn = gr.Button("Transcribe with Ichigo")
            ichigo_btn.click(
                fn=transcribe_ichigo, inputs=audio_input, outputs=ichigo_output
            )

        # Whisper Model Column
        with gr.Column():
            gr.Markdown(f"### Whisper {whisper_model_name.capitalize()}")
            whisper_output = gr.TextArea(
                label="Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            whisper_btn = gr.Button("Transcribe with Whisper")
            whisper_btn.click(
                fn=transcribe_whisper, inputs=audio_input, outputs=whisper_output
            )

        # PhoWhisper Model Column
        with gr.Column():
            gr.Markdown("### PhoWhisper Medium")
            phowhisper_output = gr.TextArea(
                label="Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            phowhisper_btn = gr.Button("Transcribe with PhoWhisper")
            phowhisper_btn.click(
                fn=transcribe_phowhisper, inputs=audio_input, outputs=phowhisper_output
            )

    custom_css = """
        .gradio-container {
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        .output-text {
            font-size: 16px;
            padding: 10px;
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        .transcribe-button {
            margin-top: 10px;
        }
    """
    gr.HTML(f"<style>{custom_css}</style>")

if __name__ == "__main__":
    interface.queue()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
