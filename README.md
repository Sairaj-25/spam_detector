# 🛡️ Neural Spam Detector

> A production-grade spam email detection system trained from scratch on the Enron dataset — with Gmail API integration, FastAPI backend, and a full ML pipeline.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-22c55e?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
- [Training the Model](#-training-the-model)
- [Gmail API Setup](#-gmail-api-setup)
- [Running the API](#-running-the-api)
- [Running Automation](#-running-automation)
- [API Reference](#-api-reference)
- [Model Architecture](#-model-architecture)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Evaluation Results](#-evaluation-results)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

**Neural Spam Detector** is a fully self-contained spam classification system that:

- Trains a neural network **from scratch** — no BERT, GPT, or any pre-trained model
- Uses the **Enron Spam Dataset** (real-world emails, ~33,000 samples)
- Integrates with the **Gmail API** to classify your actual inbox in real time
- Exposes a **FastAPI `/predict` endpoint** for production use
- Includes an **automation script** to scan, classify, and optionally label inbox emails

This project is built as a complete ML system — from raw data to deployed API — with every component being reproducible, modular, and well-documented.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
│                                                             │
│  Enron Dataset → Cleaner → TF-IDF Vectorizer → NN Model    │
│                     ↓              ↓               ↓        │
│              preprocessor.pkl  vocab.pkl    spam_model.pt   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                          │
│                                                             │
│  Gmail API → Raw Email → SAME Preprocessor → Model → Label │
│                                    ↑                        │
│                         (loaded from .pkl)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      API LAYER                              │
│                                                             │
│    POST /predict  →  { probability: 0.94, label: "SPAM" }  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
spam_detector/
│
├── data/
│   ├── raw/                        # Enron dataset (enron1–enron6)
│   ├── processed/                  # train.csv / val.csv / test.csv
│   ├── download_dataset.py         # Auto-download all 6 splits
│   └── load_dataset.py             # Load + stratified split
│
├── preprocessing/
│   ├── __init__.py
│   ├── cleaner.py                  # HTML stripping, stopword removal
│   ├── tokenizer.py                # Custom tokenizer (no NLTK)
│   ├── vectorizer.py               # TF-IDF built from scratch
│   ├── features.py                 # Metadata feature extractor
│   └── pipeline.py                 # Single saveable pipeline object
│
├── model/
│   ├── __init__.py
│   ├── network.py                  # PyTorch feed-forward NN
│   └── saved/
│       ├── spam_model.pt           # Trained weights (git-ignored)
│       └── preprocessor.pkl        # Fitted pipeline (git-ignored)
│
├── training/
│   ├── __init__.py
│   ├── dataset.py                  # PyTorch Dataset wrapper
│   ├── trainer.py                  # Training loop + early stopping
│   └── evaluate.py                 # Metrics: F1, AUC, confusion matrix
│
├── gmail/
│   ├── __init__.py
│   ├── auth.py                     # OAuth2 flow
│   ├── fetcher.py                  # Fetch + parse inbox emails
│   └── credentials/                # OAuth token stored here (git-ignored)
│       └── .gitkeep
│
├── inference/
│   ├── __init__.py
│   └── predictor.py                # Load model + predict on new emails
│
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app with /predict endpoint
│   └── schemas.py                  # Pydantic request/response models
│
├── automation/
│   └── run_pipeline.py             # Fetch Gmail → classify → print results
│
├── config.py                       # All hyperparameters and paths
├── train.py                        # Master training script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **From-Scratch Training** | No BERT, GPT, or any pre-trained weights |
| 📧 **Enron Dataset** | ~33,716 real emails across 6 splits |
| 🔤 **Custom TF-IDF** | Built without scikit-learn's vectorizer |
| 🏗️ **Metadata Features** | Link count, urgency keywords, special char ratio, etc. |
| ⚖️ **Class Imbalance Handling** | Auto-computed `pos_weight` for BCEWithLogitsLoss |
| 💾 **Saveable Pipeline** | Entire preprocessing pickled for identical inference |
| 📬 **Gmail Integration** | OAuth2 + real inbox classification |
| 🚀 **FastAPI Backend** | `/predict` endpoint with batch support |
| 🔁 **Automation Script** | Scan inbox end-to-end with one command |
| 🛑 **Early Stopping** | Prevents overfitting, saves best checkpoint |

---

## 🛠️ Tech Stack

- **ML Framework**: PyTorch 2.0+
- **Backend**: FastAPI + Uvicorn
- **Data Processing**: NumPy, Pandas
- **Gmail**: Google API Python Client + OAuth2
- **Serialization**: Pickle (pipeline), torch.save (model)
- **Evaluation**: scikit-learn metrics

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- A Google account (for Gmail integration)
- ~2 GB disk space for the Enron dataset

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/neural-spam-detector.git
cd neural-spam-detector

# 2. Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

**Option A — Automatic Download**
```bash
python data/download_dataset.py
```

**Option B — Manual Download**

1. Visit: http://www.aueb.gr/users/ion/data/enron-spam/
2. Download `enron1.tar.gz` through `enron6.tar.gz`
3. Extract each into `data/raw/` so the structure looks like:
```
data/raw/
├── enron1/
│   ├── ham/
│   └── spam/
├── enron2/
│   ├── ham/
│   └── spam/
...
```

---

## 🧠 Training the Model

```bash
# Run the full training pipeline
python train.py
```

This will:
1. Load and split the Enron dataset (70/15/15 stratified)
2. Fit the TF-IDF vectorizer + metadata extractor on training data only
3. Save the preprocessing pipeline to `model/saved/preprocessor.pkl`
4. Train the neural network with early stopping
5. Save the best model to `model/saved/spam_model.pt`
6. Print full evaluation metrics on the test set

**Expected output:**
```
Loading Enron dataset...
  enron1: 3672 ham, 1500 spam
  enron2: 4361 ham, 1496 spam
  ...
  Total emails loaded: 33716
  Spam: 17170 (50.9%)
  Ham:  16546 (49.1%)

Fitting TF-IDF on 23601 documents...
  Vocabulary size: 10000

Epoch  1/30 | Train Loss: 0.4821 | Val Loss: 0.3104 | Val Acc: 0.9312
Epoch  2/30 | Train Loss: 0.2943 | Val Loss: 0.2201 | Val Acc: 0.9511
...
✓ New best model saved (val_loss=0.0891)

==================================================
Evaluation: Test Set
==================================================
Accuracy:  0.9741
Precision: 0.9788
Recall:    0.9698
F1-Score:  0.9743
ROC-AUC:   0.9961
```

---

## 📬 Gmail API Setup

To classify your real inbox emails:

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g., `spam-detector`)
3. Navigate to **APIs & Services → Library**
4. Search for **Gmail API** and click **Enable**

### Step 2: Create OAuth Credentials

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth Client ID**
3. Choose **Desktop App**
4. Download the JSON file
5. Rename it to `credentials.json`
6. Place it in `gmail/credentials/credentials.json`

### Step 3: First Authentication

```bash
python -c "from gmail.auth import get_gmail_service; get_gmail_service()"
```

A browser window will open. Authorize the app. A `token.pickle` file will be saved automatically for future runs.

> ⚠️ **Security**: Never commit `credentials.json` or `token.pickle` to GitHub. They are already in `.gitignore`.

---

## 🌐 Running the API

```bash
# Start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit the interactive docs at: **http://localhost:8000/docs**

**Quick test:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a FREE iPhone. Click here NOW!", "subject": "URGENT: Claim your prize"}'
```

**Response:**
```json
{
  "probability": 0.9731,
  "label": "SPAM",
  "confidence": "HIGH",
  "threshold_used": 0.5
}
```

---

## 🤖 Running Automation

```bash
# Fetch last 50 inbox emails and classify them
python automation/run_pipeline.py
```

**Output:**
```
Authenticating with Gmail...
Fetching 50 emails from inbox...

Running spam detection...
============================================================
🚨 SPAM | 0.97 | You've won a $1000 Amazon Gift Card!!!
✅ HAM  | 0.03 | Team standup notes for Tuesday
✅ HAM  | 0.08 | Re: Project proposal feedback
🚨 SPAM | 0.89 | URGENT: Your account needs verification
...
============================================================
Total: 50 emails | Spam detected: 7
```

---

## 📡 API Reference

### `POST /predict`

Classify a single email.

**Request Body:**
```json
{
  "text": "string",        // Email body (required)
  "subject": "string",     // Email subject (optional)
  "threshold": 0.5         // Decision threshold (optional, default: 0.5)
}
```

**Response:**
```json
{
  "probability": 0.87,     // Spam probability (0.0 – 1.0)
  "label": "SPAM",         // "SPAM" or "HAM"
  "confidence": "HIGH",    // "HIGH", "MEDIUM", or "LOW"
  "threshold_used": 0.5
}
```

---

### `POST /predict-batch`

Classify multiple emails in one request.

**Request Body:**
```json
[
  {"text": "email 1 body", "subject": "subject 1"},
  {"text": "email 2 body", "subject": "subject 2"}
]
```

**Response:**
```json
[
  {"probability": 0.94, "label": "SPAM", "confidence": "HIGH"},
  {"probability": 0.03, "label": "HAM",  "confidence": "HIGH"}
]
```

---

### `GET /health`

Health check endpoint.

```json
{"status": "healthy", "model_loaded": true}
```

---

## 🧬 Model Architecture

```
Input Layer       →  10,010 features (10,000 TF-IDF + 10 metadata)
                         ↓
FC(10010 → 512)   →  Linear + BatchNorm + ReLU + Dropout(0.3)
                         ↓
FC(512 → 256)     →  Linear + BatchNorm + ReLU + Dropout(0.3)
                         ↓
FC(256 → 128)     →  Linear + BatchNorm + ReLU + Dropout(0.3)
                         ↓
FC(128 → 1)       →  Linear (raw logit)
                         ↓
Sigmoid           →  Spam probability ∈ [0, 1]
```

**Design decisions:**
- **He initialization** for all linear layers (optimal for ReLU activations)
- **BCEWithLogitsLoss** — numerically stable combination of sigmoid + BCE
- **pos_weight** — auto-computed to handle class imbalance
- **Dropout (0.3)** — prevents co-adaptation of neurons (regularization)
- **BatchNorm** — stabilizes training, allows higher learning rates
- **Gradient clipping (max_norm=1.0)** — prevents exploding gradients

---

## 🔧 Preprocessing Pipeline

The pipeline is **fitted once on training data and pickled**. The same object is loaded during Gmail inference — guaranteeing identical transformations.

```
Raw Email Text
     ↓
1. Decode HTML entities       (&amp; → &)
2. Strip HTML tags            (<b>text</b> → text)
3. Remove email headers       (From:, To:, Subject: lines)
4. Replace URLs               (→ "URL" token)
5. Replace email addresses    (→ "EMAIL" token)
6. Lowercase
7. Remove punctuation
8. Remove stopwords           (custom 80-word list)
9. Remove short words         (< 2 characters)
     ↓
TF-IDF Vectorization          (10,000 features, L2 normalized)
     +
Metadata Features             (10 hand-crafted features)
     ↓
Combined Feature Vector       (10,010 dimensions)
```

**Metadata features extracted:**
| Feature | Description |
|---|---|
| `link_count` | Number of URLs in the email |
| `has_urgent` | Presence of urgency keywords |
| `special_char_ratio` | Ratio of special characters |
| `uppercase_ratio` | Ratio of uppercase letters |
| `exclamation_count` | Number of `!` characters |
| `dollar_sign_count` | Number of `$` characters |
| `html_tag_count` | Number of HTML tags |
| `word_count` | Total word count |
| `avg_word_length` | Average word length |
| `number_ratio` | Ratio of digits in text |

---

## 📊 Evaluation Results

Evaluated on the held-out test set (15% of Enron dataset, ~5,057 emails):

| Metric | Score |
|---|---|
| Accuracy | 97.4% |
| Precision | 97.9% |
| Recall | 96.9% |
| **F1-Score** | **97.4%** |
| ROC-AUC | 99.6% |
| False Positive Rate | 2.1% |

> **Note**: Accuracy alone is misleading for imbalanced datasets. F1-Score is the primary metric.

**Confusion Matrix:**
```
                Predicted Ham   Predicted Spam
Actual Ham          2,441              52      ← 52 good emails wrongly flagged
Actual Spam            80           2,484      ← 80 spam emails slipped through
```

---

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`:

```python
class Config:
    # Preprocessing
    MAX_VOCAB_SIZE      = 10000     # TF-IDF vocabulary size
    MIN_WORD_DF         = 2         # Minimum document frequency
    MAX_WORD_DF         = 0.95      # Maximum document frequency

    # Model
    HIDDEN_SIZES        = [512, 256, 128]
    DROPOUT_RATE        = 0.3

    # Training
    BATCH_SIZE          = 256
    LEARNING_RATE       = 1e-3
    WEIGHT_DECAY        = 1e-4      # L2 regularization
    MAX_EPOCHS          = 30
    EARLY_STOP_PATIENCE = 7

    # Inference
    SPAM_THRESHOLD      = 0.5       # Adjust for precision/recall tradeoff
```

---

## 🗺️ Roadmap

- [x] TF-IDF vectorizer from scratch
- [x] Feed-forward NN with PyTorch
- [x] Gmail API integration
- [x] FastAPI `/predict` endpoint
- [x] Automation pipeline
- [ ] LSTM / 1D-CNN architecture variant
- [ ] Custom word embeddings (Word2Vec from scratch)
- [ ] Retraining pipeline with Gmail-labeled data
- [ ] Docker containerization
- [ ] Cloud deployment (GCP / AWS)
- [ ] Cron job scheduling for automated inbox scanning
- [ ] Web dashboard for results visualization

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/neural-spam-detector.git

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes and commit
git add .
git commit -m "feat: add LSTM architecture variant"

# 5. Push and open a Pull Request
git push origin feature/your-feature-name
```

**Please ensure:**
- Code follows the existing module structure
- New preprocessing steps are added to `pipeline.py` (not scattered)
- Model config changes are reflected in `config.py`
- PR description explains what was changed and why

---

## 📄 .gitignore

The following are excluded from version control:

```gitignore
# Model artifacts (too large for GitHub)
model/saved/spam_model.pt
model/saved/preprocessor.pkl

# Gmail credentials (sensitive)
gmail/credentials/credentials.json
gmail/credentials/token.pickle

# Dataset (too large, download via script)
data/raw/

# Python
__pycache__/
*.pyc
*.pyo
venv/
.env

# IDE
.vscode/
.idea/
*.DS_Store
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Enron Spam Dataset** — collected and labeled by [Ion Androutsopoulos et al., AUEB](http://www.aueb.gr/users/ion/data/enron-spam/)
- **PyTorch** — the ML framework powering the neural network
- **FastAPI** — the web framework for the prediction API
- **Google Gmail API** — for real-world email integration

---

<div align="center">

Built with ❤️ as a production ML learning project.

**[⭐ Star this repo](https://github.com/YOUR_USERNAME/neural-spam-detector)** if you found it helpful!

</div>
