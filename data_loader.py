import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
tqdm.pandas()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    tokens = tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

# Base directories
ORIGINAL_BASE = "data"
CLEANED_BASE = "Cleaned Data"

def save_clean_df(df, original_path, filename_suffix="_clean.csv"):
    relative_dir = os.path.dirname(os.path.relpath(original_path, ORIGINAL_BASE))
    clean_dir = os.path.join(CLEANED_BASE, relative_dir)
    os.makedirs(clean_dir, exist_ok=True)
    new_path = os.path.join(clean_dir, os.path.basename(original_path).replace(".csv", filename_suffix))
    df.to_csv(new_path, index=False)
    print(f"Saved cleaned file → {new_path}")

# Process all CSV files in the data/ directory
for root, _, files in os.walk(ORIGINAL_BASE):
    for file in files:
        if file.endswith(".csv"):
            full_path = os.path.join(root, file)
            print(f"Loading {full_path}...")
            df = pd.read_csv(full_path)
            if "Resume_str" in df.columns:
                print("Preprocessing Resume_str...")
                df["clean_text"] = df["Resume_str"].progress_apply(preprocess_text)
            elif "description" in df.columns:
                print("Preprocessing description...")
                df["clean_description"] = df["description"].progress_apply(preprocess_text)

            save_clean_df(df, full_path)

print("All files cleaned and saved to Cleaned Data/")
