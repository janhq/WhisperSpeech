import os
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
from huggingface_hub import hf_hub_download
from transformers import pipeline

from config.vq_config import VQConfig
from models.factory import make_vq_model

device = "cuda" if torch.cuda.is_available() else "cpu"
Ichigo_name = "jan-hq/ichigo-quantizer:epoch_accuracy=0.95.ckpt"
model_size = "medium-vi-2d-2048c-dim64"
whisper_model_name = "medium"
language = "vi"

whisper_model = whisper.load_model(whisper_model_name)
whisper_model.to(device)


def load_model(
    ref,
    size: str = model_size,
    repo_id=None,
    filename=None,
    local_dir=None,
    local_filename=None,
):
    """Load model from file or Hugging Face Hub.

    Args:
        ref (str): Either a local path or "repo_id:filename" format
        repo_id (str, optional): Hugging Face repository ID
        filename (str, optional): Filename in the repository
        local_dir (str, optional): Local directory for downloads
        local_filename (str, optional): Direct path to local file

    Returns:
        RQBottleneckTransformer: Loaded model instance
    """
    # Parse reference string
    if repo_id is None and filename is None and local_filename is None:
        if ":" in ref:
            repo_id, filename = ref.split(":", 1)
        else:
            local_filename = ref

    # Download or use local file
    if not os.path.exists(f"{local_filename}"):
        local_filename = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=local_dir
        )

    # Load and validate spec
    spec = torch.load(local_filename)
    model_state_dict = {
        k.replace("model.", ""): v for k, v in spec["state_dict"].items()
    }
    vq_config = VQConfig()
    ichigo_model = make_vq_model(size=size, config=vq_config)
    ichigo_model.load_state_dict(model_state_dict)
    ichigo_model.eval()
    return ichigo_model


ichigo_model = load_model(ref=Ichigo_name, size=model_size)
ichigo_model.ensure_whisper(device, language)
ichigo_model.to(device)

phowhisper = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-large",
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
        language=language,
        fp16=False,
    )
    return transcribe["text"]


def transcribe_phowhisper(inputs):
    wav2, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav2 = torchaudio.functional.resample(wav2, sr, 16000)
    audio_sample = wav2.squeeze().float().numpy()
    return phowhisper(audio_sample)["text"]


with gr.Blocks(title="Ichigo Whisper Quantizer") as interface:
    gr.Markdown(
        "# Ichigo Whisper Quantizer: Enhanced Whisper Model for Low-Resource Languages Using Quantization"
    )
    gr.Markdown("Record your voice or upload audio and send it to the model.")
    gr.Markdown("Powered by [Homebrew Ltd](https://homebrew.ltd/)")
    with gr.Row():
        # Audio input section
        audio_input = gr.Audio(
            sources=["upload", "microphone"], type="filepath", label="Audio Input"
        )

    with gr.Row():
        # Ichigo Model Column
        with gr.Column():
            gr.Markdown("### Ichigo Whisper Medium")
            ichigo_output = gr.TextArea(
                label="Ichigo Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            ichigo_btn = gr.Button("Transcribe with Ichigo")
            ichigo_btn.click(
                fn=transcribe_ichigo, inputs=audio_input, outputs=ichigo_output
            )

        # Whisper Model Column
        with gr.Column():
            gr.Markdown(f"### Whisper {whisper_model_name.upper()}")
            whisper_output = gr.TextArea(
                label="Whisper Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            whisper_btn = gr.Button("Transcribe with Whisper")
            whisper_btn.click(
                fn=transcribe_whisper, inputs=audio_input, outputs=whisper_output
            )

        # PhoWhisper Model Column
        with gr.Column():
            gr.Markdown("### PhoWhisper Model")
            phowhisper_output = gr.TextArea(
                label="PhoWhisper Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            phowhisper_btn = gr.Button("Transcribe with PhoWhisper")
            phowhisper_btn.click(
                fn=transcribe_phowhisper, inputs=audio_input, outputs=phowhisper_output
            )

    # Add some styling
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
    interface.launch("0.0.0.0", 7860, share=True)
