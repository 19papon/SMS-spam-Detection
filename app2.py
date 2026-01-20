import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.set_page_config(page_title="Smart Spam Detector", page_icon="🛡️", layout="wide")
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

ps = PorterStemmer()

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

def clear_text():
    st.session_state['input_text'] = "" 
    if 'main_input' in st.session_state:
        st.session_state['main_input'] = ""

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [ps.stem(i) for i in text if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

try:
    hv = pickle.load(open('hv_vectorizer.pkl', 'rb'))
    model = pickle.load(open('online_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found!")

def update_stats(result_type, is_correction=False):
    filename = 'status.csv'
    if not os.path.isfile(filename):
        df_stats = pd.DataFrame([[0, 0, 0]], columns=['Spam', 'Ham', 'Corrections'])
        df_stats.to_csv(filename, index=False)
    
    df_stats = pd.read_csv(filename)
    if is_correction:
        df_stats['Corrections'] += 1
    else:
        if result_type == 1: df_stats['Spam'] += 1
        else: df_stats['Ham'] += 1
    df_stats.to_csv(filename, index=False)

with st.sidebar:
    st.title("Settings")
    password = st.text_input("Developer Access", type="password", help="Only for Papon")
    
    if password == "papon786":
        st.success("Developer Mode Active")
        st.markdown("---")
        st.title("📊 Personal Analytics")
        if os.path.isfile('status.csv'):
            stats = pd.read_csv(filename := 'status.csv')
            st.metric("Total Spam Detected", stats['Spam'][0])
            st.metric("Total Safe Detected", stats['Ham'][0])
            st.metric("User Corrections (Errors)", stats['Corrections'][0], delta_color="inverse")
            
            st.bar_chart(pd.DataFrame({'Count': [stats['Spam'][0], stats['Ham'][0], stats['Corrections'][0]]}, 
                                     index=['Spam', 'Ham', 'Errors']))
    else:
        st.info("System is running in Public Mode.")

st.title("🛡️ Smart AI Spam Classifier")
st.caption("Developed by Papon")
st.markdown("---")

col_header, col_clear = st.columns([8, 2])
with col_header:
    st.subheader("📩 Paste your message below")
with col_clear:
    st.button('❌ Clear Box', on_click=clear_text)

input_sms = st.text_area(
    label="Message Input",
    value=st.session_state['input_text'],
    placeholder="Enter text here...",
    height=150,
    key="main_input", 
    label_visibility="collapsed"
)

if st.button('Analyze Message', type="primary"):
    if input_sms:
        transformed = transform_text(input_sms)
        vector_input = hv.transform([transformed])
        result = model.predict(vector_input)[0]
        
        update_stats(result)
        st.session_state['last_input'] = transformed
        
        st.markdown("### Result:")
        if result == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This message looks SAFE.")
    else:
        st.warning("Please enter a message first.")

st.markdown("---")
st.markdown("### 🤖 Help the AI Learn")
st.write("If the result was wrong, please help us improve by clicking below:")
c1, c2 = st.columns(2)

with c1:
    if st.button('❌ Mark as Spam'):
        if 'last_input' in st.session_state:
            model.partial_fit(hv.transform([st.session_state['last_input']]), [1])
            pickle.dump(model, open('online_model.pkl', 'wb'))
            update_stats(None, is_correction=True) 
            st.toast("Thank you! Model updated.", icon="🔥")

with c2:
    if st.button('✔️ Mark as Safe'):
        if 'last_input' in st.session_state:
            model.partial_fit(hv.transform([st.session_state['last_input']]), [0])
            pickle.dump(model, open('online_model.pkl', 'wb'))
            update_stats(None, is_correction=True) 
            st.toast("Thank you! Model updated.", icon="🌱")



