# IchigoWhisper 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65713d70f56f9538679e5a56/11T2v8rzhkK3OLWIl0c62.png)

Ichigo Whisper is a compact (22M parameters), open-source speech tokenizer for the `Whisper-medium` model, designed to enhance performance on multilingual with minimal impact on its original English capabilities. Unlike models that output continuous embeddings, Ichigo Whisper compresses speech into discrete tokens, making it more compatible with large language models (LLMs) for immediate speech understanding.

This speech tokenizer has been trained on over ~400 hours of English data and ~1000 hours of Vietnamese data.

Ichigo Whisper is a key component of the [Ichigo v0.5 family](https://github.com/janhq/ichigo).

For more details, please refer to our official [blog post](https://huggingface.co/homebrewltd/Ichigo-whisper-v0.1).


## Installation
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
python demo/inference.py -i path/to/your/audio.wav 

# Test example 
# python demo/inference.py -i demo/samples/test.wav
```

## Demo
```python
python demo/app.py
```
