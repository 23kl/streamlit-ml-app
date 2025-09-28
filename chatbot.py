import os
import re
import tempfile
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Voice + Translation
from deep_translator import GoogleTranslator
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI Configuration
st.set_page_config(
    page_title="Kisaan Mitra - Farmer's Herb Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
st.title("🌿 किसान मित्र - Farmer's Herb Assistant")
st.caption("Your Digital Companion for Herb Farming & Regulations")

# Sidebar - Simplified
with st.sidebar:
    st.header("📚 User Manual")
    with st.expander("🎯 How to Use", expanded=True):
        st.markdown("""
        1. Select your **language**  
        2. Choose **input method** (Text or Voice)  
        3. Ask your question about:  
           - 🌱 Cultivation techniques  
           - 🌧️ Climate & soil requirements  
           - 📜 Government regulations  
           - 💰 Market information  
        """)

    with st.expander("🎤 Voice Input Guide", expanded=True):
        st.markdown("""
        - Click **Start Recording**  
        - Speak clearly  
        - Click **Stop Recording**  
        - Your speech will be converted to text automatically  
        """)

# 🌐 Preferences Section
st.subheader("🌐 Choose Preferences")
col1, col2 = st.columns(2)

with col1:
    languages = {
        "English": "en",
        "Hindi": "hi",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Tamil": "ta"
    }
    user_lang = st.selectbox(
        "Language",
        list(languages.keys()),
        index=0,
        help="Select your preferred language"
    )

with col2:
    mode = st.radio(
        "Input Method",
        ["Text", "Voice"],
        horizontal=True
    )

st.divider()

# ❓ Question Input Section
st.subheader("❓ Ask Your Question")

if mode == "Text":
    query_text = st.text_input(
        "Type your question",
        placeholder="e.g., How to grow Ashwagandha in Maharashtra?"
    )

elif mode == "Voice":
    st.caption("🎤 Speak your question below")
    audio_data = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key="recorder",
        use_container_width=True
    )
    if audio_data and "bytes" in audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmpfile:
            tmpfile.write(audio_data["bytes"])
            tmpfile_path = tmpfile.name
        wav_path = tmpfile_path.replace(".webm", ".wav")
        subprocess.run(["ffmpeg", "-y", "-i", tmpfile_path, "-ar", "16000", "-ac", "1", wav_path], check=True)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        try:
            query_text = recognizer.recognize_google(audio, language=languages[user_lang])
            st.success(f"🗣️ You said: {query_text}")
        except:
            query_text = ""
            st.error("❌ Could not understand the audio. Please try again.")
    else:
        query_text = ""

# Load vector store
@st.cache_resource
def load_vector_store():
    loader = TextLoader("app/india_herbs_regions_soil_climate_rules.csv")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_vector_store()

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Create RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Clean text
def clean_text(text: str) -> str:
    text = re.sub(r"[*_#`]", "", text)
    text = text.replace("•", " ").replace("-", " ")
    return text.strip()

# Answer button
st.divider()
col1, col2, col3 = st.columns([1,2,1])
with col2:
    get_answer_btn = st.button("🌱 Get Expert Answer", use_container_width=True)

# Process Query
if get_answer_btn:
    if not query_text.strip():
        st.warning("⚠️ Please enter or speak a question first.")
    else:
        st.info(f"👤 Your Question: {query_text}")

        query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)
        answer = qa.run(query_in_english)
        answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)
        answer_clean = clean_text(answer_translated)

        st.success("🌿 Expert Answer")
        st.write(answer_clean)

        st.caption("💡 Powered by AI Agriculture Expert")

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "🌾 <b>Kisaan Mitra</b> - Empowering Farmers with AI Technology<br>"
    "<small>Always consult local agricultural experts for specific regional advice</small>"
    "</p>",
    unsafe_allow_html=True
)


