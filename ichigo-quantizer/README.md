1. Create virtual enviroment
   ```bash
   python -m venv quant
   source quant/bin/activate
   ```
2. Clone the repository and install requirement packages
   ```bash
   git clone https://github.com/janhq/WhisperSpeech.git
   cd WhisperSpeech/ichigo-quantizer
   pip install -r requirements.txt
   ```
3. Login Huggingface CLI and WandB
   ```bash
   huggingface-cli login
   wandb login
   ```
4. Modify config and run scripts
   ```bash
   sh scripts/train_bud500.sh
   ```