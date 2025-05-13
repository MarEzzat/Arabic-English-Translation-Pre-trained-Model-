# Arabic-English Translation Using Pretrained Model

## Overview
This project implements an Arabic-to-English neural machine translation (NMT) system using a fine-tuned MarianMT model (`Helsinki-NLP/opus-mt-ar-en`). It processes Arabic text, handles linguistic nuances (e.g., diacritics, orthographic variations), and translates to English. The project leverages Hugging Face Transformers and a custom Arabic-English dataset for training and evaluation.

### Key Features
- **Preprocessing**: Cleans Arabic text (removes diacritics, normalizes characters) and prepares data for training.
- **Fine-tuning**: Adapts the pre-trained MarianMT model to a custom dataset.
- **Evaluation**: Assesses performance using BLEU and ChrF metrics, with qualitative translation outputs.
- **Inference**: Supports translation of long texts via chunking and batch processing.
- **Tools**: Python, Hugging Face Transformers, Pandas, Scikit-learn, Evaluate.

## Requirements
- **Dependencies**:
  ```bash
  pip install transformers datasets evaluate pandas sklearn torch
  ```
- **Dataset**: Custom Arabic-English parallel corpus (`ara_eng.txt`)
- **Hardware**: GPU recommended (CUDA-compatible) for faster training; CPU supported.

## Download Dataset
- **Source**: [ara_eng.txt](https://raw.githubusercontent.com/SamirMoustafa/nmt-with-attention-for-ar-to-en/master/ara_.txt)
- **Command**:
  ```bash
  wget https://raw.githubusercontent.com/SamirMoustafa/nmt-with-attention-for-ar-to-en/master/ara_.txt -O ara_eng.txt
  ```

## Project Structure
- `ara_eng.txt`: Arabic-English parallel corpus (tab-separated).
- `main.py` (or equivalent): Main script for preprocessing, fine-tuning, evaluation, and inference.
- `./results/`: Directory for saved fine-tuned model and tokenizer.
- `./logs/`: Directory for training logs.

## Usage

### Colab Notebook
- Run the entire project (preprocessing, fine-tuning, evaluation, inference) in Google Colab:
  - [Arabic-English Translation Notebook](https://colab.research.google.com/drive/1HJcRkdiP3TeCRW-suzGK0jrPMwoakQ9x?usp=sharing)
    
### 1. Preprocessing
- **Purpose**: Clean and prepare the dataset.
- **Steps**:
  - Loads `ara_eng.txt` into a Pandas DataFrame.
  - Cleans Arabic text (removes diacritics, normalizes characters like `ى` → `ي`, keeps `؟!.،`).
  - Splits data: 80% training, 20% testing (random state: 42).
  - Tokenizes using MarianTokenizer (max length: 256 tokens).

### 2. Fine-Tuning
- **Purpose**: Adapt MarianMT model to the custom dataset.
- **Model**: `Helsinki-NLP/opus-mt-ar-en` (Transformer-based Seq2Seq).
- **Config**:
  - Epochs: 10
  - Batch size: 6 (train/eval)
  - Logging: Every 50 steps (`./logs`)
  - Output: `./results`
  - No checkpoints (`save_strategy="no"`)
- **Run**:
  ```bash
  python main.py --train
  ```
  (Assumes `main.py` includes the training logic; adjust as needed.)
- **Output**: Fine-tuned model and tokenizer saved to `./results`.

### 3. Evaluation
- **Purpose**: Assess model performance.
- **Metrics**:
  - **BLEU**: Measures n-gram overlap for fluency.
  - **ChrF**: Character n-gram F-score, robust for Arabic.
- **Process**:
  - Generates translations (beam search, 10 beams, length penalty 1.5).
  - Computes BLEU/ChrF scores using `evaluate` library.
  - Prints samples (Arabic input, predicted/true English).
- **Run**:
  ```bash
  python main.py --evaluate
  ```
- **Example Output**:
  ```
  Arabic:    هل ترغب في الرقص معي؟
  Predicted: Would you like to dance with me?
  True:      Would you like to dance with me?
  ChrF Score: <score>
  BLEU Score: <score>
  ```

### 4. Inference
- **Purpose**: Translate new Arabic text.
- **Method**: Splits long texts into 300-char chunks, processes in batches (size: 6), uses beam search (5 beams).
- **Run**:
  ```bash
  python main.py --infer
  ```
  Prompts for Arabic text input and outputs the English translation.

## Model Details
- **Model**: MarianMT (`Helsinki-NLP/opus-mt-ar-en`)
- **Tokenizer**: MarianTokenizer
- **Settings**:
  - Max length: 256 tokens
  - Beam search: 10 (eval), 5 (inference)
  - Length penalty: 1.5 (eval)
  - Device: GPU (CUDA) or CPU
- **Dataset**:
  - Source: `ara_eng.txt` (GitHub)
  - Split: 80% train, 20% test
  - Preprocessing: Diacritic removal, character normalization
