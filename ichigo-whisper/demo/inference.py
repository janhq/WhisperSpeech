import sys
import warnings
from pathlib import Path

import torch
import torchaudio

warnings.filterwarnings("ignore", category=FutureWarning)

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils import load_model

ichigo_model = load_model(
    ref="homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth",
    size="merge-medium-vi-2d-2560c-dim64",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
ichigo_model.ensure_whisper(device, language="demo")
ichigo_model.to(device)

inputs = "demo/samples/male_calm.wav"

wav, sr = torchaudio.load(inputs)
transcribe = ichigo_model.inference(wav.to(device))

print(transcribe[0].text)
