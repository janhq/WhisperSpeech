# IchigoWhisper 

## Installation
1. Create virtual enviroment
   ```bash
   python -m venv ichigo-whisper
   source ichigo-whisper/bin/activate
   ```
2. Clone the repository and install requirement packages (Python 3.11)
   ```bash
   git clone https://github.com/janhq/WhisperSpeech.git
   cd WhisperSpeech/ichigo-whisper
   pip install -r requirements.txt
   ```
3. Login Huggingface CLI and WandB
   ```bash
   huggingface-cli login
   wandb login
   ```
## Training
Modify config and run scripts
```bash
sh scripts/train_multi.sh
```

## Testing
After training, modify inference config and run scripts
```bash
sh scripts/test.sh
```

## Inference
```bash
python demo/inference.py --input path/to/your/audio.wav
```

## Demo
```python
python demo/app.py
```
