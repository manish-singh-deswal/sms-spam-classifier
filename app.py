import pickle
import string 
import nltk
import  streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

# Cache NLTK setup to prevent downloading on every interaction
@st.cache_resource
def setup_nltk():
    # Setup NLTK Resources 
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    return PorterStemmer(), set(stopwords.words("english"))


ps, stop_words = setup_nltk()


# 1.Text Pre-Processing function
def transform_text(text : str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    stemmed_words = [
            ps.stem(word)
            for word in tokens
            if word.isalnum() and word not in stop_words
        ]
    return " ".join(stemmed_words)

# 2. Load pickle files
@st.cache_resource
def load_pickle():
    
    with open("03_models/vectorizer.pkl","rb") as f:
        tfidf = pickle.load(f)
        
    with open("03_models/model.pkl","rb") as f:
        model = pickle.load(f)
        
    return tfidf, model
tfidf , model = load_pickle()

# 3. Streamlit Interface UI elements
st.title("📩 SMS / Email Spam Classifier")
st.write("Enter an SMS or Email message below to predict whether it is Spam Or Not Spam")

# take input from the User
input_sms = st.text_area("Message Content",height=140,placeholder="write your text here ....")

# 4. App logic Working 

if st.button('Analyze Message',type='primary'):
    # handle the spaces in the message
    if not input_sms.strip():
            st.warning("Please enter a message to classify.")
    else:
        # Step1 : Text Preprocess
        transformed_sms = transform_text(input_sms)
        
        # Step2 : Vectorise Convert Text to Numbers
        vector_input =  tfidf.transform([transformed_sms])
        
        # Step 3 : Predication
        prediction = model.predict(vector_input)[0]
        
        # Step 4 : Display Result 
        if prediction == 1:
                st.error("🚨 **Prediction:** This message is **SPAM**.")
        else:
                st.success("✅ **Prediction:** This message is **NOT SPAM (Ham)**.")
                
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector_input)[0]
            confidence = probs[prediction] * 100
            st.caption(f"Confidence score: {confidence:.2f}%")        
                
        