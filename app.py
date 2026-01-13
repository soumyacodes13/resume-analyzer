import streamlit as st
import pickle
import re
import docx
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity

# 1. CONFIG
st.set_page_config(page_title="AI Resume Screening", layout="wide")

# 2. LOAD RESOURCES
@st.cache_resource
def load_resources():
    try:
        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        le = pickle.load(open('encoder.pkl', 'rb'))
        ats = pickle.load(open('ats_scorer.pkl', 'rb'))
        prototypes = pickle.load(open('prototypes.pkl', 'rb'))
        return clf, tfidf, le, ats, prototypes
    except FileNotFoundError:
        return None, None, None, None, None

clf, tfidf, le, ats_model, prototypes = load_resources()

# 3. UTILS
def clean_text(txt):
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    return txt.lower()

def extract_text(file):
    try:
        if file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages])
        elif file.name.endswith('.docx'):
            doc = docx.Document(file)
            return " ".join([p.text for p in doc.paragraphs])
        elif file.name.endswith('.txt'):
            return file.read().decode('utf-8')
    except:
        return ""

def calculate_scores(text, category):
    # Retrieve the "Master Profile" for the predicted category
    if category not in prototypes:
        return 0, 0, 0
    
    master_profile = prototypes[category]
    cleaned_resume = clean_text(text)
    
    # 1. Cosine Similarity
    vecs = tfidf.transform([cleaned_resume, master_profile])
    cosine_sim = cosine_similarity(vecs[0], vecs[1])[0][0]
    
    # 2. Keyword Match
    res_tokens = set(cleaned_resume.split())
    mp_tokens = set(master_profile.split())
    keyword_match = len(res_tokens.intersection(mp_tokens)) / len(mp_tokens) if mp_tokens else 0
    
    # 3. AI Prediction
    try:
        ml_score = ats_model.predict([[cosine_sim, keyword_match]])[0]
    except:
        ml_score = 0
    
    # 4. Fallback Logic (Prevent 0 Scores)
    # If the AI predicts extremely low but similarity is okay, fallback to math
    if ml_score < 10:
        final_score = cosine_sim * 100
    else:
        final_score = ml_score

    # Visual Scaling (Raw cosine sim is usually low, e.g. 0.4, we map it to 0-100 scale)
    if final_score < 1: # If it's 0.85 style
        final_score *= 100
        
    return round(final_score, 1), round(cosine_sim*100, 1), round(keyword_match*100, 1)

# 4. MAIN APP
def main():
    st.title("📄 AI Resume Classifier & ATS Scorer")
    st.markdown("Powered by `AzharAli05` (Classification) & `0xnbk` (Scoring)")
    
    if not clf:
        st.error("⚠️ Models missing! Run `train_model.py` then `train_ats_model.py`.")
        st.stop()
        
    file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])
    
    if file:
        text = extract_text(file)
        if len(text) > 20:
            # Predict Category
            clean = clean_text(text)
            vec = tfidf.transform([clean])
            cat_id = clf.predict(vec)[0]
            category = le.inverse_transform([cat_id])[0]
            
            # Predict Score
            ats_score, raw_sim, key_match = calculate_scores(text, category)
            
            # Display
            st.success(f"### Predicted Role: {category}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("ATS Score (AI)", f"{ats_score}%")
            col2.metric("Content Match", f"{raw_sim}%")
            col3.metric("Keyword Overlap", f"{key_match}%")
            
            st.progress(min(ats_score/100, 1.0))
            
            if ats_score > 75: 
                st.balloons()
                st.info("Great match!")
            elif ats_score < 40:
                st.warning("Low match. Try adding more relevant keywords.")

            with st.expander("Show Extracted Text"):
                st.text(text)
        else:
            st.warning("Could not extract text. File might be an image/scan.")

if __name__ == "__main__":
    main()