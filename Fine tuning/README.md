# Fine-tuning Methods
1. 基於指令的微調 (Prompt Tuning)
2. 單任務微調
3. 參數高效微調 (PEFT)

# Preparation of your data
1. Collenct instruction-response pairs
2. Concatenate pairs
3. Tokenize and Truncate
4. Split into train and test

# Training Process
## What's going on?
- Add training data.
- Calculate loss
- Backprop through model
- Update weights
  
## Hyperparameters
- Learning Rate
- Learning rate scheduler
- Optimizer hyperparameters


# 縮減運算量與記憶體 Techniques
## Quantization
目的是減少並加速運算資源，將64bit的位元quantize成16bits，

## LoRA
透過Matrix decomposition的方式，將參數縮減
<img width="834" alt="螢幕擷取畫面 2024-09-17 105235" src="https://github.com/user-attachments/assets/247d8a82-92b0-4db0-a5a8-afefdfbdd6dd">

## QLoRA
