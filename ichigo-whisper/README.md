<div align="center">

# 🍰 Ichigo Whisper

[**About**](#about) | [**Demo**](#demo) | [**Training**](#training) | [**Testing**](#testing) | [**Inference**](#inference)

</div>

| ![WER](https://github.com/janhq/WhisperSpeech/blob/main/ichigo-whisper/assets/wer.png) | 
|:--:| 
| Ichigo Whisper WER on Vietnamese and English |

## About 
Ichigo Whisper is a compact (22M parameters), open-source speech tokenizer for the `whisper-medium` model, designed to enhance performance on multilingual with minimal impact on its original English capabilities. Unlike models that output continuous embeddings, Ichigo Whisper compresses speech into discrete tokens, making it more compatible with large language models (LLMs) for immediate speech understanding.

This speech tokenizer has been trained on over ~400 hours of English data and ~1000 hours of Vietnamese data.

Ichigo Whisper is a key component of the [Ichigo v0.5 family](https://github.com/janhq/ichigo).

For more details, please refer to our official [blog post](https://huggingface.co/homebrewltd/Ichigo-whisper-v0.1).


## Get Started

### Installation
1. Create virtual enviroment
   ```bash
   # venv
   python -m venv ichigo-whisper
   source ichigo-whisper/bin/activate

   # conda 
   conda create -n ichigo-whisper python=3.11
   conda activate ichigo-whisper                                                                                                                                                             
   ```

2. Clone the repository and install requirement packages (Python 3.11)
   ```bash
   git clone https://github.com/janhq/WhisperSpeech.git
   cd WhisperSpeech/ichigo-whisper
   pip install -r requirements.txt
   ```

3. Login Huggingface CLI and WandB (Optional for training)
   ```bash
   huggingface-cli login
   wandb login
   ```

### Training

Modify config and run scripts

```bash
sh scripts/train_multi.sh
```

### Testing

After training, modify inference config and run scripts

```bash
sh scripts/test.sh
```

### Inference

```bash
python demo/inference.py -i path/to/your/audio.wav 

# Test example 
# python demo/inference.py -i demo/samples/test.wav
```

### Demo

```python
python demo/app.py
```

You can try the demo directly at [Ichigo-Whisper](https://ichigo-whisper.homebrew.ltd/)

# Citation
```
@article{IchigoWhisper 2024,
  title={IchigoWhisper},
  author={Homebrew Research},
  year=2024,
  month=December},
  url={https://huggingface.co/homebrewltd/Ichigo-whisper}
```

# Acknowledgement

- **[WhisperSpeech](https://github.com/collabora/WhisperSpeech)**

- **[Whisper](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)**

- **[Vivoice](https://huggingface.co/datasets/capleaf/viVoice)**

- **[LibriTTS]**