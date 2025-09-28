import os
import re
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Farmer's Herb Chatbot", page_icon="🌿")
st.title("🌿 Farmer’s Herb & Regulation Chatbot (Free TTS)")

# Language selector
languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Tamil": "ta"
}
user_lang = st.selectbox("🌐 Select your language:", list(languages.keys()), index=0)

# Load vector store
@st.cache_resource
def load_vector_store():
    loader = TextLoader("india_herbs_regions_soil_climate_rules.csv")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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

# Clean text for TTS
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*_#`]", "", text)
    text = text.replace("•", " ").replace("-", " ")
    text = text.replace(":", ": ")
    return text.strip()

# Convert text to MP3 bytes using gTTS
def gtts_to_bytes(text: str, lang: str) -> BytesIO:
    tts = gTTS(text=text, lang=lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Input method
mode = st.radio("Choose Input Method:", ["Text", "Voice"])
query_text = ""

if mode == "Text":
    query_text = st.text_input("Ask your question about herbs & regulations:")

elif mode == "Voice":
    st.write("🎙️ Speak your query")
    audio_data = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        key="recorder",
        format="wav"
    )

    if audio_data and "bytes" in audio_data:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_data["bytes"])
            wav_path = tmpfile.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        try:
            query_text = recognizer.recognize_google(audio, language=languages[user_lang])
            st.write("🗣️ You said:", query_text)
        except sr.UnknownValueError:
            st.error("❌ Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"❌ Could not request results; {e}")

# Run query
if st.button("Get Answer"):
    if query_text.strip() == "":
        st.warning("Please enter or speak a question.")
    else:
        with st.spinner("Thinking..."):
            # Translate query → English
            query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)

            # Get answer
            answer = qa.run(query_in_english)

            # Translate back to user language
            answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)

            # Show text
            st.success(answer_translated)

            # Clean text
            tts_text = clean_text_for_tts(answer_translated)

            # Generate audio
            audio_bytes = gtts_to_bytes(tts_text, languages[user_lang])

            # Play audio
            st.audio(audio_bytes, format="audio/mp3")


