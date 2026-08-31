import pickle
import random
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

# Random Messages Button
# 1. Sample Message Repository
SAMPLE_MESSAGES = {
    "Spam": [
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Claim call 09061701461.",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt CLAIM to 81010.",
        "Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question.",
        "Congratulations! Your mobile number was awarded £5,000 cash bonus. Call 08712460324 to claim immediately.",
    ],
    "Ham": [
        "Hey, are you free this evening? Let's catch up and grab dinner around 7 PM.",
        "Can you send me the lecture notes for modern physics when you get home?",
        "I will be reaching the lab in 10 minutes. Please keep the apparatus ready.",
        "Don't forget to submit the assignment before the midnight deadline!",
    ],
}

# 2. Initialize Session State
if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# 3. Simulator Function
def simulate_user(category: str = None):
    if category in SAMPLE_MESSAGES:
        # Pick random message from selected category
        chosen_message = random.choice(SAMPLE_MESSAGES[category])
    else:
        # Pick completely random message across all categories
        all_messages = SAMPLE_MESSAGES["Spam"] + SAMPLE_MESSAGES["Ham"]
        chosen_message = random.choice(all_messages)

    st.session_state.input_text = chosen_message


# 4. Simulation Controls in the UI
st.write("🎲 **Simulate Incoming Messages:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎲 Random (Any)", use_container_width=True):
        simulate_user()

with col2:
    if st.button("🚨 Random Spam", use_container_width=True):
        simulate_user("Spam")

with col3:
    if st.button("✅ Random Ham", use_container_width=True):
        simulate_user("Ham")

# 5. Connected Text Area
input_sms = st.text_area(
    "Message Content",
    value=st.session_state.input_text,
    height=140,
    placeholder="Write your text here or click a simulation button above...",
)


# 5. App logic Working 

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
                
        