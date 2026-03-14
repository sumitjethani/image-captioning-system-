# Image-to-Caption Generator with LSTM

A multimodal deep learning system that generates natural language descriptions for images using a **Seq2Seq architecture** with **Attention mechanism**. Built with PyTorch and deployed as an interactive Gradio app.

---

## Live Demo

Try the live demo here: [Image-to-Caption Generator](https://huggingface.co/spaces/Sumit-Jethani/Image-to-Caption-Generator-with-LSTM)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## Overview

This project implements a **Neural Storyteller** - an image captioning model that takes an image as input and generates a meaningful natural language caption. The pipeline combines:

- **ResNet50** (pretrained CNN) for extracting 2048-dimensional image feature vectors
- **LSTM with Bahdanau Attention** for generating captions word by word
- **Beam Search** and **Greedy Search** for inference
- **Gradio** for interactive web deployment

---

## Architecture

```
Image → ResNet50 (frozen) → 2048-dim feature vector
                                      ↓
                          LSTM hidden state (init_h, init_c)
                                      ↓
              [<start>] → Embedding → LSTMCell + Attention → word1
                word1  → Embedding → LSTMCell + Attention → word2
                  ...                                          ...
                wordN  → Embedding → LSTMCell + Attention → [<end>]
```

### Key Components

| Component | Details |
|-----------|---------|
| Feature Extractor | ResNet50 pretrained on ImageNet, last layer removed |
| Encoder output | 2048-dim feature vector per image |
| Word Embedding | `nn.Embedding(vocab_size, 256)` |
| Attention | Bahdanau (additive) attention, dim=256 |
| Decoder | `nn.LSTMCell(2304, 512)` — input = embed(256) + context(2048) |
| Output layer | `nn.Linear(512, vocab_size)` |
| Dropout | 0.5 during training |

---

## Dataset

- **Flickr30k** — 31,000 images with 5 human-written captions each (~155,000 total pairs)
- Train/Validation split: 80% / 20%
- Vocabulary size: ~8,000–10,000 words (frequency threshold ≥ 5)

---

## Project Structure

```
├── app.py                   # Gradio app for deployment
├── best_model_val_loss.pth  # Trained model weights (see download below)
├── vocab.pkl                # Saved vocabulary object
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sumitjethani/image-captioning-system-
cd image-captioning-system-

# Install dependencies
pip install -r requirements.txt
```

---

## Model Weights

The trained model file (`best_model_val_loss.pth`, ~185MB) is too large to store on GitHub. Download it directly from Hugging Face:

**Direct download link:**
```
https://huggingface.co/spaces/Sumit-Jethani/Image-to-Caption-Generator-with-LSTM/resolve/main/best_model_val_loss.pth
```

Or download via Python:
```python
import urllib.request

url = "https://huggingface.co/spaces/Sumit-Jethani/Image-to-Caption-Generator-with-LSTM/resolve/main/best_model_val_loss.pth"
urllib.request.urlretrieve(url, "best_model_val_loss.pth")
print("Model downloaded successfully!")
```

After downloading, place `best_model_val_loss.pth` in the root directory of the project.

---

## Usage

### Run the Gradio App locally

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

### Generate a caption in Python

```python
import torch
import pickle
from app import generate_caption_beam_search  # or greedy

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load model
checkpoint = torch.load("best_model_val_loss.pth", 
                        map_location="cpu", 
                        weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate caption
caption = generate_caption_beam_search(model, image_path, vocab, beam_width=5)
print("Generated caption:", caption)
```

---

### Inference Methods

| Method | Description |
|--------|-------------|
| Greedy Search | Picks highest probability word at each step. Fast but suboptimal. |
| Beam Search (k=3) | Maintains top-3 candidate sequences. Better quality. |
| Beam Search (k=5) | Maintains top-5 candidate sequences. Best quality. |

---

## Technologies Used

- **PyTorch** — model building and training
- **TorchVision** — ResNet50 pretrained model and image transforms
- **NLTK** — BLEU score evaluation
- **Gradio** — interactive web app
- **Hugging Face Spaces** — cloud deployment
- **Flickr30k** — training dataset (via Kaggle)
- **Matplotlib** — loss curve visualization
- **Pickle** — feature and vocabulary caching

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 128 |
| Embedding Dim | 256 |
| Decoder Dim | 512 |
| Attention Dim | 256 |
| Dropout | 0.5 |
| Learning Rate | 3e-4 |
| Epochs | 15 |
| Beam Width | 3 |
| Vocab Freq Threshold | 5 |

---

## Author

**Sumit Jethani**
- GitHub: [sumitjethani](https://github.com/sumitjethani/image-captioning-system-)
- Hugging Face: [Sumit-Jethani](https://huggingface.co/spaces/Sumit-Jethani/Image-to-Caption-Generator-with-LSTM)

---

## Acknowledgements

- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) — Xu et al. (2015) — attention mechanism paper this project is based on
- [Flickr30k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k) — Young et al.
- [ResNet50](https://arxiv.org/abs/1512.03385) — He et al. (2015)
