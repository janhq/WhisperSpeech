import sys
import warnings
import argparse
from pathlib import Path

import torch
import torchaudio

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Ichigo-Whisper model"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="demo/samples/male_calm.wav",
        help="Path to input audio file (default: demo/samples/male_calm.wav)",
    )
    return parser.parse_args()


project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from ichigo_whisper.demo.utils import load_model


def main():
    args = parse_args()

    ichigo_model = load_model(
        ref="homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth",
        size="merge-medium-vi-2d-2560c-dim64",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ichigo_model.ensure_whisper(device, language="demo")
    ichigo_model.to(device)

    try:
        wav, sr = torchaudio.load(args.input)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        transcribe = ichigo_model.inference(wav.to(device))
        print(transcribe[0].text)
    except Exception as e:
        print(f"Error processing audio file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
