import pandas as pd
import pickle
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import re
import time

def train_ats_scorer():
    # 1. Load Dependencies
    print("Loading TF-IDF Vectorizer (from Step 1)...")
    try:
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    except FileNotFoundError:
        print("ERROR: 'tfidf.pkl' not found. Run 'train_model.py' first!")
        exit()

    # 2. Load ATS Dataset (0xnbk)
    print("Loading 0xnbk/resume-ats-score-v1-en...")
    try:
        ds = load_dataset("0xnbk/resume-ats-score-v1-en")
        df = pd.DataFrame(ds['train'])
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # 3. Pre-Process
    res_col = 'text'
    score_col = 'ats_score'
    cat_col = 'original_label'

    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df.dropna(subset=[score_col, res_col], inplace=True)

    # 4. Generate Training Prototypes
    print("Generating Training Prototypes...")
    # Group resumes by label to simulate "Job Descriptions"
    train_prototypes = df.groupby(cat_col)[res_col].apply(lambda x: ' '.join(x)).to_dict()

    # Optimization: Pre-calculate vectors
    print("Pre-calculating vectors...")
    proto_vectors = {}
    proto_tokens = {}
    
    for cat, text in train_prototypes.items():
        proto_vectors[cat] = tfidf.transform([text])
        proto_tokens[cat] = set(re.findall(r'\w+', text.lower()))

    # 5. Feature Engineering
    print("Calculating features...")
    cosine_sims = []
    keyword_matches = []

    for i, row in enumerate(df.itertuples()):
        text = str(getattr(row, res_col))
        cat = getattr(row, cat_col)
        
        if cat in proto_vectors:
            # Feature 1: Similarity
            vec = tfidf.transform([text])
            target_vec = proto_vectors[cat]
            sim = cosine_similarity(vec, target_vec)[0][0]
            
            # Feature 2: Keyword Match
            tokens = set(re.findall(r'\w+', text.lower()))
            target_tokens = proto_tokens[cat]
            match = len(tokens.intersection(target_tokens)) / len(target_tokens) if target_tokens else 0
        else:
            sim = 0
            match = 0
            
        cosine_sims.append(sim)
        keyword_matches.append(match)

    df['cosine_sim'] = cosine_sims
    df['keyword_match'] = keyword_matches

    # 6. Train Regressor
    print("Training ATS Regressor...")
    X = df[['cosine_sim', 'keyword_match']]
    y = df[score_col]

    reg = GradientBoostingRegressor()
    reg.fit(X, y)

    # 7. Save
    pickle.dump(reg, open('ats_scorer.pkl', 'wb'))
    print("SUCCESS: 'ats_scorer.pkl' saved.")

if __name__ == "__main__":
    train_ats_scorer()