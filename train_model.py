import pandas as pd
import pickle
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def train_classifier():
    # 1. Load Dataset (AzharAli05)
    print("Loading AzharAli05/Resume-Screening-Dataset...")
    try:
        ds = load_dataset("AzharAli05/Resume-Screening-Dataset")
        df = pd.DataFrame(ds['train'])
        print(f"Loaded {len(df)} resumes.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # 2. Setup Columns
    # Based on your dataset check: Text='Resume', Label='Role'
    text_col = 'Resume'
    label_col = 'Role'

    # 3. Cleaning Function
    def clean_resume(txt):
        cleanText = re.sub(r'http\S+\s', ' ', str(txt))
        cleanText = re.sub(r'RT|cc', ' ', cleanText)
        cleanText = re.sub(r'#\S+\s', ' ', cleanText)
        cleanText = re.sub(r'@\S+', '  ', cleanText)
        cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', ' ', cleanText)
        cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
        cleanText = re.sub(r'\s+', ' ', cleanText)
        return cleanText

    print("Cleaning data...")
    df['cleaned_resume'] = df[text_col].apply(clean_resume)

    # 4. Generate & Save Prototypes (Crucial for App)
    print("Generating Master Profiles (Prototypes)...")
    # We combine all resumes for a specific role to create a "Master Profile"
    prototypes = df.groupby(label_col)['cleaned_resume'].apply(lambda x: ' '.join(x)).to_dict()
    pickle.dump(prototypes, open('prototypes.pkl', 'wb'))

    # 5. Encoding Labels
    le = LabelEncoder()
    df['Category_ID'] = le.fit_transform(df[label_col])

    # 6. Vectorizing
    print("Vectorizing...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf.fit(df['cleaned_resume'])
    requiredText = tfidf.transform(df['cleaned_resume'])

    # 7. Training
    print("Training Classifier...")
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(requiredText, df['Category_ID'])

    # 8. Saving Models
    print("Saving models...")
    pickle.dump(clf, open('clf.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb')) 
    pickle.dump(le, open('encoder.pkl', 'wb'))
    print("SUCCESS: Classification models + Prototypes saved.")

if __name__ == "__main__":
    train_classifier()