<div align="center">

# 🍰 Ichigo-Whisper.
<a href=''><img src='https://img.shields.io/badge/Project-Blog-Green'></a>
<a href='https://ichigo-whisper.homebrew.ltd/'><img src='https://img.shields.io/badge/Project-Demo-violet'></a>
<a href='https://arxiv.org/pdf/2410.15316'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/homebrewltd/Ichigo-whisper-v0.1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

[**About**](#about) | [**Demo**](#demo) | [**Model Summary**](#model-summary) | [**Training**](#training)


  <img src="assets/ichigowhisper.png" width="400"/>
  <p><small>Homebrewed ASR quantizer model</a></small></p>
</div>

## About

Ichigo-Whisper is a compact (22M parameters), open-source speech tokenizer designed to enhance the performance of the `Whisper-medium` model, particularly for multilingual, while maintaining strong English language capabilities.

Unlike models that output continuous embeddings, Ichigo-Whisper compresses speech into **discrete tokens**. This approach makes it more compatible with large language models (LLMs) for immediate speech understanding and downstream tasks.

<div align="center">
   <img src="assets/ichigowhisper-eval.png" width="550"/>
   <p><small>Evaluation of Ichigo Whisper's performance</small></p>
</div>

## Key Features

- Only 22M parameters, enabling deployment in resource-constrained environments.
- Specifically trained to improve performance on languages with limited data.
- Outputs discrete tokens, facilitating integration with LLMs.
- Trained on ~400 hours of English and ~1000 hours of Vietnamese data, demonstrating strong performance in both languages.
- Part of a larger family of models for multilingual speech processing.

## Model Summary

### Architecture

Ichigo-Whisper's architecture is inspired by the WhisperVQ model from [WhisperSpeech](https://github.com/collabora/WhisperSpeech). It is a quantizer built on top of the Whisper-medium model, transforming continuous audio embeddings into discrete codebook entries. This quantization process allows for more efficient integration with LLMs, enabling direct speech understanding without the need for intermediate text representation.

### Codebook Initialization

We introduce a method for initializing the codebook weights in the VQ model. Instead of random initialization, we leverage the pre-trained weights from the WhisperVQ 7-language model. We then duplicate these codebooks and introduce small random noise to each copy. After training, we merge the original WhisperVQ 7-language codebooks back into the model.

<div align="center">

  <img src="assets/ichigowhisper-mergecode.png" width="550"/>
  <p><small>Codebook initialization of Ichigo Whisper</a></small></p>
</div>


**Codebook Expansion Workflow**:

```plaintext
# 1. Initial State
Codebook 512:  [512 codes + 1 mask token]
[C1 C2 C3 ... C512 M]

Codebook 2048: [2048 codes + 1 mask token]
[D1 D2 D3 ... D2048 M]

# 2. Remove Mask Token from 512
Codebook 512 (without mask):
[C1 C2 C3 ... C512]  # 512 codes

Codebook 2048 (keeps mask):
[D1 D2 D3 ... D2048 M]  # 2049 codes

# 3. Create New Empty Codebook
New Size = 512 + 2049 = 2561 codes
[_ _ _ ... _ _ _]  # 2561 empty slots

# 4. Merge Process
Step 2: Copy 2048+mask first
[D1 D2 D3 ... D2048 M | _ _ _ ... _ _ _ _ ]
 |----2049 codes----| |-----512 slots-----|

Step 2: Copy 512 codes after
[D1 D2 D3 ... D2048 M | C1 C2 C3 ... C512 |]
 |----2049 codes----| |-----512 codes-----|
```

For further details on ablation studies related to codebook initialization, please refer to this [GitHub issue](https://github.com/janhq/ichigo/issues/144).

### Two-Phase Training Methodology

We employ a two-phase training strategy to optimize Ichigo-Whisper's performance:

*   **Phase 1:** We train the model using a KL divergence loss against the output of the Whisper-medium model. This phase establishes a strong foundation and aligns the quantizer with the original model's representations.
*   **Phase 2:** Recognizing that solely relying on Whisper-medium's output can limit performance, we introduce further training in this phase.
*   **Data Mixing:** We mix Vietnamese and English data in a ratio of approximately 7:3 during training. This helps maintain English capabilities while significantly enhancing Vietnamese performance.

## How to Get Started

### Installation
1. Create virtual enviroment (optional for source)
   ```bash
   # venv
   python -m venv ichigo-whisper
   source ichigo-whisper/bin/activate

   # conda 
   conda create -n ichigo-whisper python=3.11
   conda activate ichigo-whisper                                                                                                                                                             
   ```

2. Clone the repository and install requirement packages
   ```bash
   git clone https://github.com/janhq/WhisperSpeech.git
   cd WhisperSpeech/ichigo-whisper

   # from wheel package
   pip install -e .

   # from source 
   pip install -r requirements.txt
   cd src/ichigo-whisper
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

# Example 
# python demo/inference.py -i demo/samples/test.wav
```

### Demo

```python
python demo/app.py
```

## Join Us

🍰 Ichigo Whisper is an open research project. We're looking for collaborators, and will likely move towards crowdsourcing speech datasets in the future. 

## Acknowledgement

- [WhisperSpeech](https://github.com/collabora/WhisperSpeech): Text-to-speech model for synthetic audio generation
- [Gradio](https://www.gradio.app/): A user-friendly library for building Ichigo-Whisper demo

You can try the demo directly in [here.](https://ichigo-whisper.homebrew.ltd/)

# Citation
```
@article{IchigoWhisper-2024,
  title={Ichigo Whisper},
  author={Homebrew Research},
  year=2024,
  month=December},
  url={https://huggingface.co/homebrewltd/Ichigo-whisper-v0.1}
```

# Acknowledgement

- **[WhisperSpeech](https://github.com/collabora/WhisperSpeech)**

- **[Whisper](https://github.com/openai/whisper)**

- **[Vivoice](https://huggingface.co/datasets/capleaf/viVoice)**

- **[LibriTTS-R](https://huggingface.co/datasets/parler-tts/libritts_r_filtered)**
