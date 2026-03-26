# data/load_dataset.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_enron_dataset(raw_data_dir: str = "raw") -> pd.DataFrame:
    """
    Loads the Enron spam dataset from emails.csv
    
    Expected columns: Message ID, Subject, Message, Spam/Ham, Date
    Label values    : 'spam' → 1,  'ham' → 0
    """
    csv_path = os.path.join(raw_data_dir, "emails.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {csv_path}\n"
            "   Please place emails.csv inside data/raw/"
        )

    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)

    # ── 1. Show raw info ──────────────────────────────────────────────
    print(f"\nRaw dataset:")
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {df.columns.tolist()}")
    print(f"  Labels  : {df['Spam/Ham'].value_counts().to_dict()}")

    # ── 2. Drop completely empty rows ─────────────────────────────────
    df = df.dropna(subset=['Spam/Ham'])

    # ── 3. Create binary label ────────────────────────────────────────
    df['label'] = df['Spam/Ham'].str.strip().str.lower().map({
        'spam': 1,
        'ham' : 0
    })

    # Warn if any labels failed to map
    unmapped = df['label'].isna().sum()
    if unmapped > 0:
        print(f"  ⚠️  {unmapped} rows had unrecognized labels — dropped.")
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # ── 4. Combine Subject + Message into single text field ───────────
    # WHY: The model sees one string — subject carries strong spam signals
    # e.g. "URGENT: Claim your prize" → subject alone screams spam
    df['Subject'] = df['Subject'].fillna('').astype(str).str.strip()
    df['Message'] = df['Message'].fillna('').astype(str).str.strip()

    df['text'] = (
        'Subject: ' + df['Subject'] +
        '\n\n'      + df['Message']
    )

    # ── 5. Drop emails with almost no content ─────────────────────────
    df = df[df['text'].str.len() > 20].reset_index(drop=True)

    # ── 6. Keep only what we need ─────────────────────────────────────
    df = df[['text', 'label']].copy()

    # ── 7. Shuffle ────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── 8. Summary ────────────────────────────────────────────────────
    spam_count = df['label'].sum()
    ham_count  = (df['label'] == 0).sum()

    print(f"\n✅ Dataset loaded successfully!")
    print(f"  Total emails : {len(df):,}")
    print(f"  Spam  (1)    : {spam_count:,}  ({spam_count/len(df)*100:.1f}%)")
    print(f"  Ham   (0)    : {ham_count:,}  ({ham_count/len(df)*100:.1f}%)")

    # Class imbalance warning
    ratio = spam_count / ham_count
    if ratio < 0.4 or ratio > 2.5:
        print(f"\n  ⚠️  Class imbalance detected (spam/ham ratio = {ratio:.2f})")
        print(f"      We'll handle this with pos_weight in the loss function.")

    return df


def split_dataset(df: pd.DataFrame):
    """
    Stratified 70 / 15 / 15 split.
    Stratified = spam % is same in train, val, and test.
    """
    # Split 1: train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size  = 0.30,
        random_state = 42,
        stratify   = df['label']      # ← maintains spam ratio
    )

    # Split 2: val (15%) vs test (15%) from the temp 30%
    val_df, test_df = train_test_split(
        temp_df,
        test_size    = 0.50,
        random_state = 42,
        stratify     = temp_df['label']
    )

    print(f"\nStratified splits:")
    print(f"  {'Split':<8} {'Emails':>8} {'Spam%':>8} {'Ham%':>8}")
    print(f"  {'-'*36}")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        spam_pct = split['label'].mean() * 100
        ham_pct  = 100 - spam_pct
        print(f"  {name:<8} {len(split):>8,} {spam_pct:>7.1f}% {ham_pct:>7.1f}%")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # ── Load ──────────────────────────────────────────────────────────
    df = load_enron_dataset(raw_data_dir="raw")

    # ── Split ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = split_dataset(df)

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("processed", exist_ok=True)
    train_df.to_csv("processed/train.csv", index=False)
    val_df.to_csv(  "processed/val.csv",   index=False)
    test_df.to_csv( "processed/test.csv",  index=False)

    print(f"\n✅ Splits saved to data/processed/")
    print(f"   train.csv → {len(train_df):,} rows")
    print(f"   val.csv   → {len(val_df):,} rows")
    print(f"   test.csv  → {len(test_df):,} rows")
    print(f"\n🚀 Next step: python preprocessing/pipeline.py")