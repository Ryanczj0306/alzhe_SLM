# alzhe_SLM
# Alzheimer’s Disease Fine-tuning Corpus Generator

This repository contains a **one-stop Python script** (`kaggle_dataset.py`) that

1. **Downloads** the public Kaggle dataset *“Alzheimer’s Disease Dataset”*  
2. **Cleans** and **filters** the tabular data  
3. **Converts** each structured row into a natural-language *prompt/response* pair  
4. **Exports** a ready-to-use `jsonl` file (`alz_finetune.jsonl`) for **instruction-tuning or SFT** of
   Small / Large Language Models (SLMs & LLMs).

---

## Table of Contents
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Script Arguments](#script-arguments)
- [How the Prompt Is Composed](#how-the-prompt-is-composed)
- [Fine-tuning Example (🤗 Transformers)](#fine-tuning-example-🤗-transformers)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

| Step | Description | Output |
|------|-------------|--------|
| 1    | **Kaggle API fetch** with built-in credentials check | `data_alz/*.zip` |
| 2    | Automatic extraction | CSV file(s) |
| 3    | Lightweight cleaning  (drop constant columns & NaNs) | `pandas.DataFrame` |
| 4    | Robust prompt builder (handles missing columns) | prompt string |
| 5    | Binary diagnosis → `"Yes"` / `"No"` response | response string |
| 6    | Final corpus saved as **line-delimited JSON** | `alz_finetune.jsonl` |

---

## Folder Structure
├── kaggle_dataset.py        # main script
├── requirements.txt         # pip dependencies
├── data_alz/                # auto-created; zipped & extracted data live here
└── alz_finetune.jsonl       # output (created after first run)

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| **Python** | ≥ 3.8 | tested on 3.11 |
| **pip** | latest | `python -m pip install -U pip` |
| **Kaggle API** | 1.6.6+ | included in `requirements.txt` |

### Configure Kaggle credentials

1. Log in to <https://www.kaggle.com/settings/account>  
2. Click **“Create New API Token”** → download `kaggle.json`  
3. Move it to `~/.kaggle/kaggle.json` and run  
   ```bash
   chmod 600 ~/.kaggle/kaggle.json

## Quick Start
# clone your repo
git clone https://github.com/<your-username>/alz-research.git
cd alz-research

# install dependencies
python -m venv .venv && source .venv/bin/activate   # optional virtualenv
pip install -r requirements.txt

# run the pipeline
python kaggle_dataset.py