# alzheimer_finetune_prep.py
import os, json, zipfile, shutil, re
import pandas as pd
from tqdm import tqdm

# ---------- Step ①: Download from Kaggle ----------
DATA_DIR = "data_alz"
os.makedirs(DATA_DIR, exist_ok=True)

print("⬇️  downloading from Kaggle ...")
os.system(f"kaggle datasets download -d rabieelkharoua/alzheimers-disease-dataset -p {DATA_DIR} --force")

# ---------- Step ②: Extract archive ----------
zip_path = [f for f in os.listdir(DATA_DIR) if f.endswith('.zip')][0]
with zipfile.ZipFile(os.path.join(DATA_DIR, zip_path), 'r') as zf:
    zf.extractall(DATA_DIR)

# ---------- Step ③: Read CSV ----------
csv_path = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(DATA_DIR, csv_path))

print("Rows before cleaning:", len(df))

# ---------- Step ④: Lightweight cleaning ----------
# 1) Drop all-null or constant columns
for col in df.columns:
    if df[col].nunique() <= 1:
        df.drop(columns=col, inplace=True)

# 2) Drop rows with missing values
df = df.dropna()
print("Rows after cleaning:", len(df))

# ---------- Step ⑤: Build prompt / response ----------
def row2prompt(row: pd.Series) -> str:
    """Convert a structured row into a natural‑language prompt"""
    prompt_parts = [
        f"Age: {row['Age']}",
        f"Gender: {'Female' if row['Gender']==1 else 'Male'}",
        f"EducationLevel: {row['EducationLevel']}",
        f"BMI: {row['BMI']}",
        f"Smoking: {'Yes' if row['Smoking']==1 else 'No'}",
        f"FamilyHistoryAlzheimers: {'Yes' if row['FamilyHistoryAlzheimers']==1 else 'No'}",
        f"MMSE: {row['MMSE']}",
        f"Forgetfulness: {'Yes' if row['Forgetfulness']==1 else 'No'}",
        # Add more features here if needed ...
    ]
    return (
        "Below is structured information for an older adult patient. "
        "Based on known Alzheimer’s risk indicators, answer whether this patient "
        "is diagnosed with Alzheimer’s disease (answer only: 'Yes' or 'No').\n\n"
        + "\n".join(prompt_parts)
    )

def label2response(label):
    return "Yes" if label==1 else "No"

records = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row2prompt(row)
    response = label2response(row['Diagnosis'])
    records.append({"prompt": prompt, "response": response})

# ---------- Step ⑥: Save JSONL ----------
out_path = "alz_finetune.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅  Generated fine‑tuning corpus {out_path}  —— {len(records)} samples in total")